from base_proposal.affordance.get_affordance import get_affordance_point


cell_size = 0.05
map_size = (203, 203)


def get_base(
    occupancy_2d_map, target, instruction, R, T, fx, fy, cx, cy, destination, K=3
):
    affordance_point, affordance_pixel = get_affordance_point(
        target, instruction, R, T, fx, fy, cx, cy, occupancy_2d_map, destination
    )
    print(f"affordance_point: {affordance_point}")
    return (affordance_point[0], affordance_point[1])
