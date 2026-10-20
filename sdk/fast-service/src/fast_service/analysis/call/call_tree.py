from typing import Optional


class NodeStatistic:

    occurrence_frequency: int

    approximate_start_point: float

    total_delay: float
    delay_list: list[float]

    communication_delay_list: list[float]

    def __init__(
        self,
        occurrence_frequency: int,
        approximate_start_point: float,
        total_delay: float,
        delay_list: list[float],
        communication_delay_list: list[float],
    ):
        self.approximate_start_point = approximate_start_point
        self.occurrence_frequency = occurrence_frequency
        self.total_delay = total_delay
        self.delay_list = delay_list
        self.communication_delay_list = communication_delay_list


class FunctionNode:
    node_id: str
    function_name: str

    parent_node: Optional["FunctionNode"]
    children_dict: dict[str, "FunctionNode"]
    children_list: list["FunctionNode"]

    end_time: float
    delay: float

    new_instance: bool

    statistic: Optional[NodeStatistic]

    def __init__(
        self,
        node_id: str,
        function_name: str,
        parent_node: Optional["FunctionNode"],
        end_time: float,
        delay: float,
        new_instance: bool,
    ):
        self.node_id = node_id
        self.function_name = function_name
        self.parent_node = parent_node
        self.end_time = end_time
        self.delay = delay
        self.new_instance = new_instance

        self.children_dict = {}
        self.children_list = []

        self.statistic = None

    def add_child(self, child_node: "FunctionNode"):
        self.children_dict[child_node.node_id] = child_node
        self.children_list.append(child_node)
        child_node.parent_node = self

    def get_child(self, child_id: str) -> "FunctionNode":
        return self.children_dict[child_id]

    def get_full_name(self) -> str:
        path = []
        current = self
        parent = self.parent_node
        while parent:
            path.append(current.function_name)
            current = parent
            parent = current.parent_node

        path.reverse()
        return ".".join(path)

    def sort_children(self):
        self.children_list.sort(key=lambda x: x.get_start_time())

    def is_leaf(self) -> bool:
        return len(self.children_list) == 0

    def get_start_time(self) -> float:
        return self.end_time - self.delay


def _compute_node_statistic(node_list: list[FunctionNode]) -> FunctionNode:
    num = len(node_list)
    delay_list = []
    communication_delay_list = []
    total_delay = 0
    start_point_list = []
    for node in node_list:
        delay_list.append(node.delay)
        total_delay += node.delay

        if node.parent_node is None:
            start_point_list.append(0.0)
        else:
            if node.new_instance:
                communication_delay_list.append(node.delay - node.parent_node.delay)

            start_point_list.append(
                (node.get_start_time() - node.parent_node.get_start_time())
                / node.parent_node.delay
            )

    if node_list[0].parent_node is None:
        start_point = 0
    else:
        start_point_sum = 0.0
        for start_point in start_point_list:
            start_point_sum += start_point
        start_point = start_point_sum / len(start_point_list)

    statistic_node = FunctionNode(
        node_list[0].get_full_name(),
        node_list[0].function_name,
        None,
        -1,
        total_delay / num,
        node_list[0].new_instance,
    )
    statistic_node.statistic = NodeStatistic(
        num, start_point, total_delay, delay_list, communication_delay_list
    )
    return statistic_node


def _statistic_node(node_list: list[FunctionNode]) -> FunctionNode:
    statistic_node = _compute_node_statistic(node_list)

    children_dict = {}
    for node in node_list:
        for child in node.children_list:
            if child.function_name not in children_dict:
                children_dict[child.function_name] = [child]
            else:
                children_dict[child.function_name].append(child)

    for children_list in children_dict.values():
        statistic_child = _statistic_node(children_list)
        statistic_node.add_child(statistic_child)

    statistic_node.children_list.sort(key=lambda x: x.statistic.approximate_start_point)

    return statistic_node


class FunctionCallTree:

    tree_id: str

    root_node: Optional[FunctionNode]

    def __init__(self, tree_id: str):
        self.tree_id = tree_id
        self.root_node = None

    def set_root_node(self, root_node: "FunctionNode"):
        self.root_node = root_node

    def is_empty(self) -> bool:
        return self.root_node is None

    @classmethod
    def compute_statistics(
        cls, group_name: str, tree_list: list["FunctionCallTree"]
    ) -> "FunctionCallTree":
        statistic_root = _statistic_node([_.root_node for _ in tree_list])

        statistic_tree = FunctionCallTree(group_name)
        statistic_tree.set_root_node(statistic_root)

        return statistic_tree
