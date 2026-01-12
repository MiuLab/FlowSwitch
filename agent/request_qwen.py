import os
import asyncio
from openai import OpenAI, AsyncOpenAI
import json
import re
import requests_async as arequests
# import logging
from loguru import logger

# 设置 vLLM 服务地址

async def request_qwen_chat(messages, model="qwen/qwen3-8b", temperature=0.7, top_p=None, top_k=None, min_p=None, max_tokens=None, retry=30, enable_thinking=False):
    """
    通过 vLLM OpenAI 接口调用 Qwen 模型生成对话
    :param messages: List[Dict]，chat 历史记录 [{"role": ..., "content": ...}]
    :param model: 使用的模型名称，如 "Qwen/Qwen3-8B"
    :param temperature: 生成多样性控制参数
    :param top_p: nucleus sampling 控制参数
    :param functions: 可调用的函数列表
    :return: 
        - 如果没有functions参数：返回 {"response": content} 格式
        - 如果有functions参数：返回包含function_call的消息格式
    """
    try:
        # 如果有functions参数，需要特殊处理
        processed_messages = messages

        if not enable_thinking:
            processed_messages[-1]["content"] += " /no_think"
        # 发起请求
#         response = await client.chat.completions.create(
#             model=model,
#             messages=processed_messages,
#             temperature=temperature,
#             top_p=top_p,
#             top_k=top_k,
#             min_p=min_p,
#             max_tokens=max_tokens,
#         )
#
#         # 提取内容
#         content = response.choices[0].message.content
        # url = "https://openrouter.ai/api/v1/chat/completions"
        url = "http://qwen.morris-chang.rocks/v1/chat/completions"

        response = await arequests.post(
          url=url,
          headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
          },
          data=json.dumps({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
          })
        )
        content = response.json()["choices"][0]
        # print("[Qwen-vLLM 回复]:", content[:200] + ("..." if len(content) > 200 else ""))
        
        # 如果有functions，尝试解析函数调用
        return {"response": content}

    except Exception as e:
        # logger.error(f"[Qwen-vLLM 错误] 请求失败: {e}")
        logger.error(f"Qwen-vLLM request failed: {e} with {response.json()}")
        if retry > 0:
            logger.warning(f"Retrying request after 15 seconds...")
            await asyncio.sleep(15)
            return await request_qwen_chat(messages, model, temperature, top_p, top_k, min_p, max_tokens, retry-1, enable_thinking)
        else:
            raise RuntimeError("Qwen-vLLM 请求失败")


def _build_function_system_message(functions):
    """构建包含函数信息的系统消息"""
    function_descriptions = []
    for func in functions:
        func_desc = f"Function: {func['name']}\n"
        func_desc += f"Description: {func.get('description', '')}\n"
        
        if 'parameters' in func and 'properties' in func['parameters']:
            func_desc += "Parameters:\n"
            for param_name, param_info in func['parameters']['properties'].items():
                param_desc = param_info.get('description', '')
                param_type = param_info.get('type', 'string')
                func_desc += f"  - {param_name} ({param_type}): {param_desc}\n"
        
        function_descriptions.append(func_desc)
    
    system_message = """You are a helpful assistant that can call functions when needed.

Available functions:
""" + "\n\n".join(function_descriptions) + """

When you need to call a function, use the following format in your response:
Thought: [Your reasoning about why you need to call this function]
Action: [function_name]
Action Input: [function parameters as JSON]
Response: [Your response to the user]

If you don't need to call any function, just provide a normal response with:
Thought: [Your reasoning]
Response: [Your response to the user]
"""
    
    return system_message


def _add_function_context(messages, function_system_message):
    """将函数上下文添加到消息列表中"""
    processed_messages = []
    
    # 检查是否已经有系统消息
    has_system_message = any(msg.get("role") == "system" for msg in messages)
    
    if has_system_message:
        # 如果已有系统消息，将函数信息添加到第一个系统消息中
        for msg in messages:
            if msg.get("role") == "system":
                enhanced_content = msg["content"] + "\n\n" + function_system_message
                processed_messages.append({"role": "system", "content": enhanced_content})
            else:
                processed_messages.append(msg)
    else:
        # 如果没有系统消息，添加一个新的
        processed_messages.append({"role": "system", "content": function_system_message})
        processed_messages.extend(messages)
    
    return processed_messages


def _parse_function_call(content, functions):
    """从响应内容中解析函数调用"""
    try:
        # 查找 Action 行
        action_match = re.search(r'Action:\s*(.+)', content)
        if not action_match:
            return None
        
        function_name = action_match.group(1).strip()
        
        # 查找 Action Input 行
        action_input_match = re.search(r'Action Input:\s*(.+?)(?=\nResponse:|$)', content, re.DOTALL)
        if not action_input_match:
            return None
        
        action_input_str = action_input_match.group(1).strip()
        
        # 尝试解析 JSON
        try:
            if action_input_str.startswith('{') and action_input_str.endswith('}'):
                action_input = json.loads(action_input_str)
            else:
                # 如果不是有效的JSON，尝试简单解析
                action_input = {"query": action_input_str}
        except json.JSONDecodeError:
            # 如果JSON解析失败，将整个字符串作为参数
            action_input = {"query": action_input_str}
        
        # 验证函数名是否存在
        function_names = [func["name"] for func in functions]
        if function_name not in function_names:
            print(f"[Qwen-vLLM 警告] 未知函数名: {function_name}")
            return None
        
        return {
            "name": function_name,
            "arguments": action_input
        }
        
    except Exception as e:
        print(f"[Qwen-vLLM 警告] 解析函数调用失败: {e}")
        return None


# 为了向后兼容，保留原始函数名
def request_qwen_chat_simple(messages, model="Qwen/Qwen3-8B", temperature=0.7, top_p=0.9):
    """
    简化版本，只返回文本内容
    """
    result = request_qwen_chat(messages, model, temperature, top_p)
    if isinstance(result, dict) and "response" in result:
        return result["response"]
    return str(result)
