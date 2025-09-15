import json
from loguru import logger
from utils import summarize_role, summarize_domain, MAPPING

if __name__ == "__main__":
    # summarize_role()
    roles = {}
    domains = {}
    for domain in MAPPING.keys():
        summaries = []
        logger.info(f"=====Processing domain: {domain}=====")
        for role in MAPPING[domain].keys():
            logger.info(f"    ----Summarizing role: {role}----")
            summary = summarize_role(role)
            roles[role] = summary
            summaries.append(f"Role Name: {role}\nDescription: {summary}")
        logger.info(f"    ----Summarizing domain: {domain}----")
        domain_desc = summarize_domain(domain, summaries)
        logger.info(f"===FINISHED domain: {domain}=====")
        domains[domain] = domain_desc
        break
    with open("role_desc.json", "w") as f:
        json.dump(roles, f, indent=4)
    with open("domain_desc.json", "w") as f:
        json.dump(domains, f, indent=4)
