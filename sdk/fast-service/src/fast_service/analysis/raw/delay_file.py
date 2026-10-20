class DelayRecord:
    end_time: float
    delay: float

    request_id: str

    invoker_name: str
    function_name: str

    call_id: str
    parent_call_id: str

    def __init__(
        self,
        end_time: float,
        delay: float,
        request_id: str,
        invoker_name: str,
        function_name: str,
        call_id: str,
        parent_call_id: str,
    ):
        self.end_time: float = end_time
        self.delay: float = delay
        self.request_id: str = request_id
        self.invoker_name: str = invoker_name
        self.function_name: str = function_name
        self.call_id: str = call_id
        self.parent_call_id: str = parent_call_id


class DelayFile:
    file_path: str
    instance_name: str
    content: list[DelayRecord]

    def __init__(self, instance_name: str, file_path: str):
        self.instance_name = instance_name
        self.file_path = file_path

        self.content = []
        with open(file_path, "r") as f:
            line = f.readline()
            while line:
                fields = line.strip().split(",")
                end_time = float(fields[0])
                request_id = fields[1]
                invoker_name = fields[2]
                function_name = fields[3]
                delay = float(fields[4])
                call_id = fields[5]
                parent_call_id = fields[6]
                self.content.append(
                    DelayRecord(
                        end_time,
                        delay,
                        request_id,
                        invoker_name,
                        function_name,
                        call_id,
                        parent_call_id,
                    )
                )

                line = f.readline()


class DelayFileGroup:
    group_name: str
    # instance_name -> delay
    file_dict: dict[str, list[DelayFile]]

    def __init__(self, group_name: str):
        self.group_name = group_name
        self.file_dict = {}

    def add_file(self, file: DelayFile):
        instance_name = file.instance_name
        if instance_name not in self.file_dict:
            self.file_dict[instance_name] = []
        self.file_dict[instance_name].append(file)


class DelayFileManager:

    group_dict: dict[str, DelayFileGroup]

    def __init__(self):
        self.group_dict: dict[str, DelayFileGroup] = {}

    def add_file(self, group_name: str, instance_name: str, file_path: str):
        if group_name not in self.group_dict:
            self.group_dict[group_name] = DelayFileGroup(group_name)
        self.group_dict[instance_name].add_file(DelayFile(instance_name, file_path))

    def add_file_list(
        self, group_name: str, instance_name: str, file_path_list: list[str]
    ):
        for file_path in file_path_list:
            self.add_file(group_name, instance_name, file_path)

    def add_file_groups(self, group_dict: dict[str, dict[str, list[str]]]):
        # group name -> instance name -> file path list
        for group_name, instance_dict in group_dict.items():
            for instance_name, file_path_list in instance_dict.items():
                self.add_file_list(group_name, instance_name, file_path_list)
