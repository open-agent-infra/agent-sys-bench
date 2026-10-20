from typing import Optional

from ..raw import DelayRecord, DelayFile
from ..call import FunctionNode, FunctionCallTree


def convert_record_to_node(
    record: DelayRecord, parent_node: Optional[FunctionNode]
) -> FunctionNode:
    return FunctionNode(
        node_id=record.call_id,
        function_name=record.function_name,
        parent_node=parent_node,
        end_time=record.end_time,
        delay=record.delay,
        new_instance=record.invoker_name is None,
    )


class RequestDelay:

    request_id: str

    stage_list: list[DelayRecord]
    stage_dict: dict[str, DelayRecord]

    function_tree: FunctionCallTree

    def __init__(self, first_stage: DelayRecord):
        self.request_id = first_stage.request_id
        self.stage_list = [first_stage]
        self.stage_dict = {first_stage.call_id: first_stage}
        self.function_tree = FunctionCallTree(self.request_id)

    def append(self, stage_delay: DelayRecord):
        if self.function_tree.is_empty():
            self.stage_list.append(stage_delay)
            self.stage_dict[stage_delay.call_id] = stage_delay

    def construct_function_tree(self) -> bool:
        root_node = None
        children_dict: dict[str, list[FunctionNode]] = {}
        node_dict: dict[str, FunctionNode] = {}
        for stage in self.stage_list:
            node = convert_record_to_node(stage, None)
            node_dict[stage.call_id] = node
            if stage.call_id in children_dict:
                for child_node in children_dict[stage.call_id]:
                    node.add_child(child_node)
                children_dict.pop(stage.call_id)

            if stage.parent_call_id == "None":
                if root_node is not None:
                    return False
                root_node = node
            elif stage.parent_call_id in node_dict:
                node_dict[stage.parent_call_id].add_child(node)
            else:
                if stage.parent_call_id not in children_dict:
                    children_dict[stage.parent_call_id] = []
                children_dict[stage.parent_call_id].append(node)

        if root_node is None:
            return False
        else:
            self.function_tree.set_root_node(root_node)
            for node in node_dict.values():
                node.sort_children()
            return True


class RequestDelayManager:

    group_name: str

    delay_file_list: list[DelayFile]

    request_dict: dict[str, RequestDelay]

    statistic_tree: Optional[FunctionCallTree]

    def __init__(self, group_name: str, delay_file_list: list[DelayFile]):
        self.group_name = group_name
        self.delay_file_list = delay_file_list

        self.request_dict = {}

        self.statistic_tree = None

    def _process_single_file(self, delay_file: DelayFile):
        for delay_record in delay_file.content:
            if delay_record.request_id not in self.request_dict:
                request_delay = RequestDelay(delay_record)
                self.request_dict[delay_record.request_id] = request_delay
            else:
                request_delay = self.request_dict[delay_record.request_id]
                request_delay.append(delay_record)

    def prepare_in_request(self):
        for delay_file in self.delay_file_list:
            self._process_single_file(delay_file)

        request_list = list(self.request_dict.values())
        for request_delay in request_list:
            if not request_delay.construct_function_tree():
                self.request_dict.pop(request_delay.request_id)

        self.statistic_tree = FunctionCallTree.compute_statistics(
            self.group_name, [_.function_tree for _ in request_list]
        )

    def print_flatten_operations(self):
        root_node = self.statistic_tree.root_node
        node_list: [tuple[str, float]] = []

        def collect_nodes(node: FunctionNode):
            node_list.append(
                (
                    node.function_name,
                    node.statistic.total_delay / root_node.statistic.total_delay,
                )
            )
            for child_node in node.children_list:
                collect_nodes(child_node)

        collect_nodes(root_node)

        node_list.sort(key=lambda x: x[1], reverse=True)
        for node in node_list:
            print(node)
