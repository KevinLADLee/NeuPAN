from neupan import neupan
import irsim
import numpy as np
import argparse
import csv

def main(
    env_file,
    planner_file,
    save_animation=False,
    ani_name="animation",
    full=False,
    no_display=True,
    point_vel=False,
    max_steps=1000,
    log_csv_path=None,
    reverse=False,
):
    env = irsim.make(env_file, save_ani=save_animation, full=full, display=no_display)
    neupan_planner = neupan.init_from_yaml(planner_file)

    # Resolve sample time for logging if available
    sample_time = getattr(env, "sample_time", None)
    if sample_time is None:
        world = getattr(env, "world", None)
        if world is not None:
            sample_time = getattr(world, "sample_time", None) or getattr(world, "step_time", None)

    log_file = None
    csv_writer = None
    if log_csv_path:
        try:
            log_file = open(log_csv_path, "w", newline="")
            csv_writer = csv.writer(log_file)
            csv_writer.writerow(
                [
                    "step",
                    "time",
                    "x",
                    "y",
                    "yaw",
                    "raw_vx_local",
                    "raw_vy_local",
                    "raw_omega",
                    "cmd_vx",
                    "cmd_vy",
                    "cmd_omega",
                ]
            )
        except OSError as exc:
            print(f"Failed to open log file {log_csv_path}: {exc}")
            log_file = None
            csv_writer = None

    # neupan_planner.update_adjust_parameters(q_s=0.5, p_u=1.0, eta=10.0, d_max=1.0, d_min=0.1)
    # neupan_planner.set_reference_speed(5)
    # neupan_planner.update_initial_path_from_waypoints([np.array([0, 0, 0]).reshape(3, 1), np.array([100, 100, 0]).reshape(3, 1)])

    try:
        for i in range(max_steps):

            robot_state = env.get_robot_state()
            lidar_scan = env.get_lidar_scan()

            if point_vel:
                points, point_velocities = neupan_planner.scan_to_point_velocity(robot_state, lidar_scan)
            else:
                points = neupan_planner.scan_to_point(robot_state, lidar_scan)
                point_velocities = None

            action, info = neupan_planner(robot_state, points, point_velocities)
            raw_action = np.copy(action)

            if info["stop"]:
                print("NeuPAN stops because of minimum distance")

            if info["arrive"]:
                print("NeuPAN arrives at the target")
                break

            env.draw_points(neupan_planner.dune_points, s=25, c="g", refresh=True)
            env.draw_points(neupan_planner.nrmp_points, s=13, c="r", refresh=True)
            env.draw_trajectory(neupan_planner.opt_trajectory, "r", refresh=True)
            env.draw_trajectory(neupan_planner.ref_trajectory, "b", refresh=True)

            # For omni, ir-sim expects [vx, vy, omega] in robot frame; keep as-is
            # If other kinematics, the planner already outputs the correct shape

            if csv_writer:
                time_val = i * sample_time if sample_time is not None else i
                raw_vx = float(raw_action[0, 0]) if raw_action.shape[0] > 0 else None
                raw_vy = float(raw_action[1, 0]) if raw_action.shape[0] > 1 else None
                raw_omega = float(raw_action[2, 0]) if raw_action.shape[0] > 2 else None

                cmd_vx = float(action[0, 0]) if action.shape[0] > 0 else None
                cmd_vy = float(action[1, 0]) if action.shape[0] > 1 else None
                cmd_omega = float(action[2, 0]) if action.shape[0] > 2 else None

                csv_writer.writerow(
                    [
                        i,
                        time_val,
                        float(robot_state[0, 0]),
                        float(robot_state[1, 0]),
                        float(robot_state[2, 0]),
                        raw_vx,
                        raw_vy,
                        raw_omega,
                        cmd_vx,
                        cmd_vy,
                        cmd_omega,
                    ]
                )

            env.step(action)
            env.render()

            if env.done():
                break

            if i == 0:

                if reverse:
                    # for reverse motion
                    for j in range(len(neupan_planner.initial_path)):
                        neupan_planner.initial_path[j][-1, 0] = -1
                        neupan_planner.initial_path[j][-2, 0] = neupan_planner.initial_path[j][-2, 0] + 3.14

                    env.draw_trajectory(neupan_planner.initial_path, traj_type="-k", show_direction=True)
                else:
                    env.draw_trajectory(neupan_planner.initial_path, traj_type="-k", show_direction=False)

                env.render()
    finally:
        if log_file:
            log_file.close()
            print(f"Control log saved to {log_csv_path}")

        env.end(3, ani_name=ani_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--example", type=str, default="polygon_robot", help="pf, pf_obs, corridor, dyna_obs, dyna_non_obs, convex_obs, non_obs, polygon_robot, reverse")
    parser.add_argument("-d", "--kinematics", type=str, default="diff", help="acker, diff, omni")
    parser.add_argument("-a", "--save_animation", action="store_true", help="save animation")
    parser.add_argument("-f", "--full", action="store_true", help="full screen")
    parser.add_argument("-n", "--no_display", action="store_false", help="no display")
    parser.add_argument("-v", "--point_vel", action='store_true', help="point vel")
    parser.add_argument("-m", "--max_steps", type=int, default=1000, help="max steps")
    parser.add_argument("--log_csv", type=str, default=None, help="path to save a single-run control CSV log")

    args = parser.parse_args()

    env_path_file = args.example + "/" + args.kinematics + "/env.yaml"
    planner_path_file = args.example + "/" + args.kinematics + "/planner.yaml"

    ani_name = args.example + "_" + args.kinematics + "_ani"

    reverse = (args.example == "reverse" and args.kinematics == "diff")

    main(
        env_path_file,
        planner_path_file,
        args.save_animation,
        ani_name,
        args.full,
        args.no_display,
        args.point_vel,
        args.max_steps,
        args.log_csv,
        reverse,
    )
