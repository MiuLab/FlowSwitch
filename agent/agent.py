import json
from request_llm import request_llm_vllm_chat, request_llm_vllm_completion


class WorkflowAgent:
    def __init__(self):
        self.agent_name = "Workflow Agent"
        self.system_message = """### Insturctions:
        You are an agent designed to interact with a user in a conversational manner.
        Your goal is to help the user complete their task according to different workflow SOPs.
        In order to accomplish this, you will need to understand the user's intention and determine the appropriate workflow SOP to follow.
        Specifically, given the current dialogue context, current workflwo SOP and the user's intention, you will need to decide:
        1. Whether the user's intention is aligned with the current workflow SOP.
            a. If the answer is no, does current user's intention has to do with any tasks?
                i, If yes, you will need generate a suitable search query to find the appropriate workflow SOP.
                ii, If no, you will need to respond to the user directly.
            b. If the answer is yes, you will need to respond to the user directly without any further action.
        2. Note that All you have to do is to decide which action to take, you do not need to worry about any availability of the actions/functions.
        ### Current Workflow SOPs:
        {current_workflow_sop}
        
        ### Output Format:
        Follow the below format in every response under any circumstances:
        ```json
        {{
            "action": "<search, stay>",
            "search_query": "<search query>",(empty if action is not search)
            "user_intention": "<user intention>",
            "if_out_of_scope": true/false
        }}
        ```
        ### Reponse:
        """
        self.agent_kwargs = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0,
            "max_new_tokens": 5000,
        }

    def get_response(
        self, context, current_workflow_sop, enable_thinking=False, retry=5
    ):
        system_message = self.system_message.format(
            current_workflow_sop=current_workflow_sop
        )
        messages = [{"role": "system", "content": system_message}] + context
        response = (
            request_llm_vllm_chat(
                messages, enable_thinking=enable_thinking, **self.agent_kwargs
            )
            .message.content.split("assistant:")[-1]
            .strip()
        )
        reasoning = response.split("</think>")[0].strip().replace("<think>", "").strip()
        response = (
            response.split("</think>")[-1]
            .strip()
            .replace("```json", "")
            .replace("```", "")
        )
        try:
            response = json.loads(response)
            return response
        except:
            if retry > 0:
                return self.get_response(
                    context,
                    current_workflow_sop,
                    enable_thinking=enable_thinking,
                    retry=retry - 1,
                )
            else:
                raise ValueError("Failed to parse response")

    def get_workflows(self, input, strates: str = ""):
        pass
