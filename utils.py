import os
import json
import jsonlines

MAPPING = {
    "Customer Service": {
        "restaurant_waiter": [
            "Find a restaurant",
            "Book a restaurant",
        ],
        "hotel_reception": [
            "Find a hotel",
            "Book a hotel room",
            "Book a hotel",
        ],
        "apartment_manager": [
            "Search for apartments",
            "Make an appointment to view a home",
            "Apartment fee payment",
            "Apartment fee bill inquiry",
        ],
        "gas_equipment_service": [
            "Gas repairs",
            "Gas fee inquiry and payment",
            "Feedback on Gas Service Interruption",
        ],
    },
    "Personal Assistant": {
        "medical_consultant": [
            "Make an appointment with a doctor",
            "Get diagnostic results",
        ],
        "meeting_arrangement": [
            "Launch the meeting",
            "Cancel meeting",
        ],
        "financial_assistant": [
            "Currency exchange",
            "Appointments for large withdrawals",
            "Check the balance and transfer the bank card account",
        ],
    },
    "E-tailing Recommandation": {
        "online_shopping_support": [
            "Search for products",
            "Shopping cart management",
            "Order operations",
        ],
        "computer_store_sales": [
            "Notebook purchase",
            "Laptop Returns",
        ],
    },
    "Travel&Transportation": {
        "ride_service": [
            "Initiate a ride order",
            "Initiate a ride change order",
        ],
        "driving_service": [
            "Consultation and appointment for Substitute driving service",
            "Cancel driving appointments",
            "Change surrogate driving reservation",
        ],
        "flight_inquiry": [
            "Check flights",
            "Book a flight",
        ],
        "travel_assistant": [
            "Travel guidance",
            "Weather services",
        ],
    },
    "Logistics Solutions": {
        "express_support": [
            "Express shipping",
            "Query express information",
        ],
        "moving_service": [
            "Moving Service Appointment",
            "Insurance claim",
        ],
        "food_delivery_service": ["Online surveys", "Get Food Voucher"],
    },
    "Robotic Process Automation": {
        "invoice_management": ["Invoice management", "Invoice reimbursement"],
        "mail_administration": ["Send mail", "Reply to an email"],
        "printing_service": ["File printing", "Notification of file results"],
        "attendance_arrangement": ["Attendance Anomaly Detection", "Shift Handover"],
        "seal_management": ["Seal Request", "Seal State Notification"],
        "workstation_applicant": [
            "Replacement of workstations",
            "Request the user to change the workstation",
        ],
    },
}


def load_pool(pool_name: str):
    """
    Loads a specified pool from the corresponding JSON file.

    Args:
        pool_name: The name of the pool to load (e.g., "workflow_text").

    Returns:
        A dictionary representing the loaded pool.
    """
    pool_path = f"pools/{pool_name}.json"
    if not os.path.exists(pool_path):
        raise FileNotFoundError(f"Pool file not found: {pool_path}")

    with open(pool_path, "r") as f:
        return json.load(f)


def load_bm25_retriever(pool_name: str):
    """
    Loads a BM25 retriever for a given pool.

    Args:
        pool_name: The name of the pool to use for the retriever.

    Returns:
        A BM25 retriever object.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError("Please install rank_bm25: pip install rank_bm25")

    pool = load_pool(pool_name)
    corpus = list(pool.values())
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def load_reranker():
    """
    Loads the intfloat/e5-base-v2 re-ranker model.

    Returns:
        A sentence transformer model for re-ranking.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "Please install sentence-transformers: pip install sentence-transformers"
        )

    return SentenceTransformer("intfloat/e5-base-v2")


def load_qwen_retriever():
    """
    Loads the Qwen/Qwen3-8B model as a retriever.

    Returns:
        A tuple of tokenizer and model for Qwen/Qwen3-8B.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-7B-Chat")
    return tokenizer, model


def scenario2domainrole(scenario_name):
    """
    Convert a scenario name to a domain name.

    Args:
        scenario_name: The name of the scenario.

    Returns:
        The domain name corresponding to the scenario.
    """
    for domain in MAPPING.keys():
        for role in MAPPING[domain].keys():
            if scenario_name in MAPPING[domain][role]:
                return domain, role
    return "", ""


def summarize_workflow(workflow_text):
    """
    Summarizes the workflow text using gpt-4.1.
    """
    return request_openai(
        prompt=f"""
    ### Instructions:
    1. Write a high-level description of the provided workflow without too many details.
    
    ### Workflow Text:
    {workflow_text}
    
    return your summary in the key "summary" in json format
    """
    )


def get_scenarios_by_role(role):
    for domain in MAPPING.keys():
        for r in MAPPING[domain].keys():
            if r == role:
                return MAPPING[domain][r]


def summarize_role(role):
    """
    Create role's description from all the corresponding scenarios.
    """
    scenario_names = get_scenarios_by_role(role)
    scenarios = []
    with jsonlines.open("workflow_final.jsonl", "r") as f:
        for line in f:
            if line["scenario"] in scenario_names:
                scenarios.append(line["workflow"]["summary"])

    return request_openai(
        prompt=f"""
    ### Instructions:
    1. Write a high-level description of the provided role based on the provided scenarios.
    2. The description should be concise and clear without too many details.
    
    ### Role:
    {role}
    ### Scenarios:
    {scenarios}
    
    return your summary in the key "summary" in json format
    """
    )


def get_roles_by_domain(domain):
    for domain in MAPPING.keys():
        if domain == domain:
            return list(MAPPING[domain].keys())


def summarize_domain(domain, roles=None):
    """
    Create domain's description from all the corresponding roles
    """
    role_names = get_roles_by_domain(domain)
    if roles is None:
        roles = []
        role_descs = json.load(open("role_desc.json"))
        for role in role_names:
            roles.append(f"Role Name: {role}\nDescription: {role_descs[role]}")
    roles = "\n".join(roles)
    return request_openai(
        prompt=f"""
    ### Instructions:
    1. Write a high-level description of the provided domain based on the provided roles.
    2. The description should be concise and clear without too many details.
    
    ### Domain:
    {domain}
    ### Roles:
    {roles}
    
    return your summary in the key "summary" in json format
    """
    )


def request_openai(prompt, model="gpt-4.1"):
    """
    Summarizes the workflow text using gpt-4.1.
    """
    from openai import OpenAI

    # Create a summary of the workflow
    client = OpenAI()
    summary = client.responses.create(
        model=model,
        input=prompt,
    )
    summary = json.loads(summary.output_text.replace("```json", "").replace("```", ""))
    return summary["summary"]
