from kiln_ai.datamodel import TaskRun

"""
Toy implementation: the idea here is that each stream consumed to completion will return a
TaskRun object.

The TaskRun contains a trace property, which is a list of messages in the OpenAI protocol format
and that trace can be passed into the Kiln adapter's invoke, invoke_openai_stream, invoke_ai_sdk_stream
methods to continue the conversation.

What you would presumably want to do here is store the TaskRun in a database, retrieve it and get the `trace`
and pass it into the adapter's invoke call. The TaskRun contains a bunch of additional metadata, which you may
or may not care about - so storing the trace directly is also sufficient.

Since the trace is in OpenAI format already, you don't need to convert the messages from UI -> backend either.
"""


class EphemeralStorage:
    def __init__(self):
        self.session_task_run: TaskRun | None = None

    def store_task_run(self, task_run: TaskRun):
        self.session_task_run = task_run

    def get_task_run(self) -> TaskRun | None:
        return self.session_task_run

fake_storage = EphemeralStorage()