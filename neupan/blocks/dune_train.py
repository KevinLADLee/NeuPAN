"""
DUNETrain is the class for training the DUNE model. It is used when you deploy the NeuPan algorithm on a new robot with a specific geometry. 

Developed by Ruihua Han
Copyright (c) 2025 Ruihua Han <hanrh@connect.hku.hk>

NeuPAN planner is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

NeuPAN planner is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with NeuPAN planner. If not, see <https://www.gnu.org/licenses/>.
"""

import torch
from colorama import deinit

deinit()

from torch.utils.data import Dataset, random_split, DataLoader
import cvxpy as cp
from rich.console import Console
from rich.progress import Progress
from rich.live import Live
from torch.optim import Adam
import numpy as np
from neupan.configuration import np_to_tensor, value_to_tensor, to_device
import pickle
import time
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from contextlib import nullcontext


class PointDataset(Dataset):
    def __init__(self, input_data, label_data, distance_data):
        """
        input_data: point p, [2, 1]
        label_data: mu, [G.shape[0], 1]
        distance_data: distance, scalar
        """

        self.input_data = input_data
        self.label_data = label_data
        self.distance_data = distance_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_sample = self.input_data[idx]
        label_sample = self.label_data[idx]
        distance_sample = self.distance_data[idx]

        return input_sample, label_sample, distance_sample


class DUNETrain:
    def __init__(self, model, robot_G, robot_h, checkpoint_path) -> None:

        self.G = robot_G
        self.h = robot_h
        self.model = model

        self.construct_problem()
        self.checkpoint_path = checkpoint_path

        self.loss_fn = torch.nn.MSELoss()

        self.optimizer = Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)

        # for rich progress
        self.console = Console()
        self.progress = Progress(transient=False)
        self.live = Live(self.progress, console=self.console, auto_refresh=False)

        # loss
        self.loss_of_epoch = 0
        self.loss_list = []

    def construct_problem(self):
        """
        optimization problem (10):

        max mu^T * (G * p - h)
        s.t. ||G^T * mu|| <= 1
            mu >= 0
        """
        self.mu = cp.Variable((self.G.shape[0], 1), nonneg=True)
        self.p = cp.Parameter((2, 1))  # points

        cost = self.mu.T @ (self.G.cpu() @ self.p - self.h.cpu())
        constraints = [cp.norm(self.G.cpu().T @ self.mu) <= 1]

        self.prob = cp.Problem(cp.Maximize(cost), constraints)

    def process_data(self, rand_p):
        distance_value, mu_value = self.prob_solve(rand_p)  # Adapted to be accessible
        return (
            np_to_tensor(rand_p),
            np_to_tensor(mu_value),
            value_to_tensor(distance_value),
        )
    
    @staticmethod
    def _process_data_worker(args):
        """Static method for parallel data processing"""
        rand_p, G_np, h_np = args
        
        # Reconstruct the problem for this worker
        mu = cp.Variable((G_np.shape[0], 1), nonneg=True)
        p = cp.Parameter((2, 1))
        cost = mu.T @ (G_np @ p - h_np)
        constraints = [cp.norm(G_np.T @ mu) <= 1]
        prob = cp.Problem(cp.Maximize(cost), constraints)
        
        # Solve
        p.value = rand_p
        try:
            prob.solve(solver=cp.ECOS)
        except Exception as exc:  # pragma: no cover - robustness
            # Fallback on solver failure
            return rand_p, np.zeros_like(h_np), 0.0

        mu_value = mu.value if mu.value is not None else np.zeros_like(h_np)
        distance_value = prob.value if prob.value is not None else 0.0

        return (rand_p, mu_value, distance_value)

    def generate_data_set(self, data_size=10000, data_range=[-50, -50, 50, 50], num_workers=0, log_benchmark=False):
        """
        generate dataset for training
        data_range: [low_x, low_y, high_x, high_y]
        num_workers: number of parallel workers for data generation (0 = sequential, -1 = use all CPUs)
        """

        input_data = []
        label_data = []
        distance_data = []

        rand_p = np.random.uniform(
            low=data_range[:2], high=data_range[2:], size=(data_size, 2)
        )
        rand_p_list = [rand_p[i].reshape(2, 1) for i in range(data_size)]

        # Prepare data for parallel processing
        G_np = self.G.cpu().numpy()
        h_np = self.h.cpu().numpy()
        
        # Determine number of workers
        if num_workers == -1:
            num_workers = cpu_count()
        elif num_workers == 0:
            num_workers = 0
        
        if num_workers > 0:
            # Parallel data generation
            print(f"Generating data with {num_workers} parallel workers...")
            start_time = time.time()
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                args_list = [(p, G_np, h_np) for p in rand_p_list]
                results = list(executor.map(DUNETrain._process_data_worker, args_list))
            
            elapsed_time = time.time() - start_time
            print(f"Data generation completed in {elapsed_time:.2f} seconds")
            if log_benchmark and elapsed_time > 0:
                print(f"Throughput: {len(results)/elapsed_time:.1f} samples/sec")
            
            for rand_p_result, mu_value, distance_value in results:
                input_data.append(np_to_tensor(rand_p_result))
                label_data.append(np_to_tensor(mu_value))
                distance_data.append(value_to_tensor(distance_value))
        else:
            # Sequential data generation (original method)
            print("Generating data sequentially...")
            start_time = time.time()
            
            for i, p in enumerate(rand_p_list):
                if (i + 1) % 10000 == 0:
                    print(f"Generated {i + 1}/{data_size} samples...")
                results = self.process_data(p)
                input_data.append(results[0])
                label_data.append(results[1])
                distance_data.append(results[2])
            
            elapsed_time = time.time() - start_time
            print(f"Data generation completed in {elapsed_time:.2f} seconds")

        dataset = PointDataset(input_data, label_data, distance_data)

        return dataset

    def prob_solve(self, p_value):

        self.p.value = p_value
        self.prob.solve(solver=cp.ECOS)  # distance
        # self.prob.solve()  # distance

        return self.prob.value, self.mu.value

    def start(
        self,
        data_size: int = 100000,
        data_range: list[int] = [-25, -25, 25, 25],
        batch_size: int = 256,
        epoch: int = 5000,
        valid_freq: int = 100,
        save_freq: int = 500,
        lr: float = 5e-5,
        lr_decay: float = 0.5,
        decay_freq: int = 1500,
        save_loss: bool = False,
        num_workers: int = 0,
        data_gen_workers: int = 0,
        dataloader_num_workers: int = 0,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        log_data_gen_benchmark: bool = False,
        **kwargs,
    ):

        train_dict = {
            "data_size": data_size,
            "data_range": data_range,
            "batch_size": batch_size,
            "epoch": epoch,
            "valid_freq": valid_freq,
            "save_freq": save_freq,
            "lr": lr,
            "lr_decay": lr_decay,
            "decay_freq": decay_freq,
            "robot_G": self.G,
            "robot_h": self.h,
            "model": self.model,
        }

        with open(self.checkpoint_path + "/train_dict.pkl", "wb") as f:
            pickle.dump(train_dict, f)

        print(
            f"data_size: {data_size}, data_range: {data_range}, batch_size: {batch_size}, epoch: {epoch}, valid_freq: {valid_freq}, save_freq: {save_freq}, lr: {lr}, lr_decay: {lr_decay}, decay_freq: {decay_freq}, robot_G: {self.G}, robot_h: {self.h}"
        )

        with open(self.checkpoint_path + "/results.txt", "a") as f:
            print(
                f"data_size: {data_size}, data_range: {data_range}, batch_size: {batch_size}, epoch: {epoch}, valid_freq: {valid_freq}, save_freq: {save_freq}, lr: {lr}, lr_decay: {lr_decay}, decay_freq: {decay_freq}, robot_G: {self.G}, robot_h: {self.h}\n",
                file=f,
            )

        self.optimizer.param_groups[0]["lr"] = float(lr)
        ful_model_name = None

        # Use data_gen_workers if provided, otherwise fall back to num_workers
        data_gen_workers = data_gen_workers if data_gen_workers > 0 else num_workers

        print("dataset generating start ...")
        dataset = self.generate_data_set(
            data_size,
            data_range,
            num_workers=data_gen_workers,
            log_benchmark=log_data_gen_benchmark,
        )

        train_len = int(data_size * 0.8)
        valid_len = data_size - train_len
        train, valid = random_split(dataset, [train_len, valid_len])

        # Optimize DataLoader for CPU training (best practices from PyTorch docs)
        # - num_workers: parallel data loading (CPU can use multiprocessing)
        # - persistent_workers: keep workers alive between epochs (reduces overhead)
        # - prefetch_factor: number of batches to prefetch per worker
        dataloader_kwargs = {
            "batch_size": batch_size,
            "num_workers": dataloader_num_workers,
            "persistent_workers": persistent_workers and dataloader_num_workers > 0,
            "prefetch_factor": prefetch_factor if dataloader_num_workers > 0 else None,
            "pin_memory": pin_memory,
        }
        # Remove None values
        dataloader_kwargs = {k: v for k, v in dataloader_kwargs.items() if v is not None}
        
        train_dataloader = DataLoader(train, shuffle=True, **dataloader_kwargs)
        valid_dataloader = DataLoader(valid, **dataloader_kwargs)
        
        if dataloader_num_workers > 0:
            print(f"DataLoader optimized: {dataloader_num_workers} workers, persistent={dataloader_kwargs.get('persistent_workers', False)}, prefetch={dataloader_kwargs.get('prefetch_factor', 2)}")

        print("dataset training start ...")

        with self.live:
            task = self.progress.add_task("[cyan]Training...", total=epoch)

            for i in range(epoch + 1):

                self.progress.update(task, advance=1)
                self.live.refresh()

                self.model.train(True)

                mu_loss, distance_loss, fa_loss, fb_loss = self.train_one_epoch(
                    train_dataloader, False
                )

                ml, dl, al, bl = (
                    "{:.2e}".format(mu_loss),
                    "{:.2e}".format(distance_loss),
                    "{:.2e}".format(fa_loss),
                    "{:.2e}".format(fb_loss),
                )

                if i % valid_freq == 0:
                    self.model.eval()
                    (
                        valid_mu_loss,
                        valid_distance_loss,
                        validate_fa_loss,
                        validate_fb_loss,
                    ) = self.train_one_epoch(valid_dataloader, True)

                    vml, vdl, val, vbl = (
                        "{:.2e}".format(valid_mu_loss),
                        "{:.2e}".format(valid_distance_loss),
                        "{:.2e}".format(validate_fa_loss),
                        "{:.2e}".format(validate_fb_loss),
                    )

                    self.print_loss(
                        i,
                        epoch,
                        ml,
                        dl,
                        al,
                        bl,
                        vml,
                        vdl,
                        val,
                        vbl,
                        self.optimizer.param_groups[0]["lr"],
                    )

                    with open(self.checkpoint_path + "/results.txt", "a") as f:
                        self.print_loss(
                            i,
                            epoch,
                            ml,
                            dl,
                            al,
                            bl,
                            vml,
                            vdl,
                            val,
                            vbl,
                            self.optimizer.param_groups[0]["lr"],
                            f,
                        )

                if i % save_freq == 0:
                    print("save model at epoch {}".format(i))
                    torch.save(
                        self.model.state_dict(),
                        self.checkpoint_path + "/" + "model_" + str(i) + ".pth",
                    )
                    ful_model_name = (
                        self.checkpoint_path + "/" + "model_" + str(i) + ".pth"
                    )

                if (i + 1) % decay_freq == 0:
                    self.optimizer.param_groups[0]["lr"] = (
                        self.optimizer.param_groups[0]["lr"] * lr_decay
                    )
                    print(
                        "current learning rate:", self.optimizer.param_groups[0]["lr"]
                    )

                    with open(self.checkpoint_path + "/results.txt", "a") as f:
                        print(
                            "current learning rate:",
                            self.optimizer.param_groups[0]["lr"],
                            file=f,
                        )

                self.loss_of_epoch = mu_loss + distance_loss + fa_loss + fb_loss
                self.loss_list.append(self.loss_of_epoch)

                if save_loss:
                    with open(self.checkpoint_path + "/loss.pkl", "wb") as f:
                        pickle.dump(self.loss_list, f)

        print("finish train, the model is saved in {}".format(ful_model_name))

        return ful_model_name

    def train_one_epoch(self, train_dataloader, validate=False):
        """
        loss:
            mu: mse between output mu and label mu
            objective function value (distance): mse between output distance and label distance
            fa: -mu^T * G * R^T  ==> lam^T
            fb: mu^T * G * R^T * p - mu^T * h  ==> lam^T * p + mu^T * h
        """

        mu_loss, distance_loss, fa_loss, fb_loss = 0, 0, 0, 0

        ctx = torch.no_grad() if validate else nullcontext()
        with ctx:
            for input_point, label_mu, label_distance in train_dataloader:

                if not validate:
                    self.optimizer.zero_grad()

                input_point = torch.squeeze(input_point)
                
                output_mu = self.model(input_point)
                output_mu = torch.unsqueeze(output_mu, 2)

                distance = self.cal_distance(output_mu, input_point)

                mse_mu = self.loss_fn(output_mu, label_mu)
                mse_distance = self.loss_fn(distance, label_distance)
                mse_fa, mse_fb = self.cal_loss_fab(output_mu, label_mu, input_point)

                loss = mse_mu + mse_distance + mse_fa + mse_fb

                if not validate:
                    loss.backward()
                    self.optimizer.step()

                mu_loss += mse_mu.item()
                distance_loss += mse_distance.item()
                fa_loss += mse_fa.item()
                fb_loss += mse_fb.item()

        return (
            mu_loss / len(train_dataloader),
            distance_loss / len(train_dataloader),
            fa_loss / len(train_dataloader),
            fb_loss / len(train_dataloader),
        )

    def cal_loss_fab(self, output_mu, label_mu, input_point):
        """
        calculate the loss of fa and fb

        fa: -mu^T * G * R^T  ==> lam^T
        fb: mu^T * G * R^T * p - mu^T * h  ==> lam^T * p + mu^T * h
        """

        mu1 = output_mu
        mu2 = label_mu
        ip = torch.unsqueeze(input_point, 2)
        mu1T = torch.transpose(mu1, 1, 2)
        mu2T = torch.transpose(mu2, 1, 2)

        theta = np.random.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        R = np_to_tensor(R)

        fa = torch.transpose(-R @ self.G.T @ mu1, 1, 2)
        fa_label = torch.transpose(-R @ self.G.T @ mu2, 1, 2)

        fb = fa @ ip + mu1T @ self.h
        fb_label = fa_label @ ip + mu2T @ self.h

        mse_lamt = self.loss_fn(fa, fa_label)
        mse_lamtb = self.loss_fn(fb, fb_label)

        return mse_lamt, mse_lamtb

    def cal_distance(self, mu, input_point):

        input_point = torch.unsqueeze(input_point, 2)

        temp = self.G @ input_point - self.h

        muT = torch.transpose(mu, 1, 2)

        distance = torch.squeeze(torch.bmm(muT, temp))

        return distance

    def print_loss(self, i, epoch, ml, dl, al, bl, vml, vdl, val, vbl, lr, file=None):

        if file is None:
            print(
                "Epoch {}/{}, learning rate {} \n"
                "---------------------------------\n"
                "Losses:\n"
                "  Mu Loss:          {} | Validate Mu Loss:          {}\n"
                "  Distance Loss:    {} | Validate Distance Loss:    {}\n"
                "  Fa Loss:          {} | Validate Fa Loss:          {}\n"
                "  Fb Loss:          {} | Validate Fb Loss:          {}\n".format(
                    i,
                    epoch,
                    lr,
                    str(ml).ljust(10),
                    str(vml).rjust(10),
                    str(dl).ljust(10),
                    str(vdl).rjust(10),
                    str(al).ljust(10),
                    str(val).rjust(10),
                    str(bl).ljust(10),
                    str(vbl).rjust(10),
                )
            )

        else:
            print(
                "Epoch {}/{} learning rate {} \n"
                "---------------------------------\n"
                "Losses:\n"
                "  Mu Loss:          {} | Validate Mu Loss:          {}\n"
                "  Distance Loss:    {} | Validate Distance Loss:    {}\n"
                "  Fa Loss:          {} | Validate Fa Loss:          {}\n"
                "  Fb Loss:          {} | Validate Fb Loss:          {}\n".format(
                    i,
                    epoch,
                    lr,
                    str(ml).ljust(10),
                    str(vml).rjust(10),
                    str(dl).ljust(10),
                    str(vdl).rjust(10),
                    str(al).ljust(10),
                    str(val).rjust(10),
                    str(bl).ljust(10),
                    str(vbl).rjust(10),
                ),
                file=file,
            )

    def test(self, model_pth, train_dict_kwargs, data_size_list=0, **kwargs):

        with open(train_dict_kwargs, "rb") as f:
            train_dict = pickle.load(f)

        model = to_device(train_dict["model"])
        model.load_state_dict(torch.load(model_pth))
        data_range = train_dict["data_range"]

        print("dataset generating start ...")

        max_data_size = max(data_size_list)

        start_time = time.time()
        dataset = self.generate_data_set(max_data_size, data_range)
        data_generate_time = time.time() - start_time
        print(
            "data_size:", max_data_size, "dataset generating time: ", data_generate_time
        )

        for data_size in data_size_list:
            test_dataloader = DataLoader(dataset, batch_size=data_size)

            mu_loss_list = []
            distance_loss_list = []
            fa_loss_list = []
            fb_loss_list = []
            inference_time_list = []

            for input_point, label_mu, label_distance in test_dataloader:
                average_loss_list, inference_time = self.test_one_epoch(
                    model, input_point, label_mu, label_distance, data_size
                )

                mu_loss_list.append(average_loss_list[0])
                distance_loss_list.append(average_loss_list[1])
                fa_loss_list.append(average_loss_list[2])
                fb_loss_list.append(average_loss_list[3])
                inference_time_list.append(inference_time)

            avg_mu_loss = sum(mu_loss_list) / len(mu_loss_list)
            avg_distance_loss = sum(distance_loss_list) / len(distance_loss_list)
            avg_fa_loss = sum(fa_loss_list) / len(fa_loss_list)
            avg_fb_loss = sum(fb_loss_list) / len(fb_loss_list)
            avg_inference_time = sum(inference_time_list) / len(inference_time_list)

            with open(os.path.dirname(model_pth) + "/test_results.txt", "a") as f:
                print(
                    "Model_name {}, Data_size {}, inference_time {} \n"
                    "---------------------------------\n"
                    "Losses:\n"
                    "  Mu Loss:          {} \n"
                    "  Distance Loss:    {} \n"
                    "  Fa Loss:          {} \n"
                    "  Fb Loss:          {} \n".format(
                        os.path.basename(model_pth),
                        data_size,
                        avg_inference_time,
                        str(avg_mu_loss).ljust(10),
                        str(avg_distance_loss).ljust(10),
                        str(avg_fa_loss).ljust(10),
                        str(avg_fb_loss).ljust(10),
                    ),
                    file=f,
                )

                # with open(os.path.dirname(model_pth) + '/results_dict.pkl', 'wb') as f:
                #     results_kwargs = { 'Model_name': os.path.basename(model_pth), 'Data_size': data_size, 'inference_time': sum(inference_time_list) / len(inference_time_list), 'mu_loss': sum(mu_loss_list)/ len(mu_loss_list), 'distance_loss': sum(distance_loss_list)/len(distance_loss_list), 'fa_loss': sum(fa_loss_list)/ len(fa_loss_list), 'fb_loss': sum(fb_loss_list) / len(fb_loss_list)}

                #     pickle.dump(results_kwargs, f)
        print(
            "finish test, the results are saved in {}".format(
                os.path.dirname(model_pth) + "/test_results.txt"
            )
        )

    def test_one_epoch(self, model, input_point, label_mu, label_distance, data_size):

        input_point = torch.squeeze(input_point)

        start_time = time.time()
        output_mu = model(input_point)
        inference_time = time.time() - start_time

        output_mu = torch.unsqueeze(output_mu, 2)

        distance = self.cal_distance(output_mu, input_point)

        mse_mu = self.loss_fn(output_mu, label_mu)
        mse_distance = self.loss_fn(distance, label_distance)
        mse_fa, mse_fb = self.cal_loss_fab(output_mu, label_mu, input_point)

        # loss = mse_mu.item() + mse_distance + mse_fa + mse_fb
        # average_loss_list = [mse_mu.item() / data_size, mse_distance.item() / data_size, mse_fa.item() / data_size, mse_fb.item() / data_size]

        loss_list = [mse_mu.item(), mse_distance.item(), mse_fa.item(), mse_fb.item()]

        # print('Data_size {}, inference_time {} \n'
        #             '---------------------------------\n'
        #             'Losses:\n'
        #             '  Mu Loss:          {} \n'
        #             '  Distance Loss:    {} \n'
        #             '  Fa Loss:          {} \n'
        #             '  Fb Loss:          {} \n'
        #             .format(data_size, inference_time,
        #                     str(average_loss_list[0]).ljust(10),
        #                     str(average_loss_list[1]).ljust(10),
        #                     str(average_loss_list[2]).ljust(10),
        #                     str(average_loss_list[3]).ljust(10)))
        return loss_list, inference_time
