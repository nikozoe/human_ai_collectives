#!/usr/bin/env python
"""
Script for prompt variation to get AI case solves.
"""

import random
import pickle
from time import sleep
from collections import defaultdict, Counter
from datetime import datetime
from typing import Optional
from copy import deepcopy

# Local modules
from llm_prompts import (
    get_open_ai_solve,
    get_google_ai_rest_solve,
    get_meta_llama_solve,
    get_anthropic_ai_solve,
    get_mistral_ai_solve,
)


# PARAMETERS
BASE_DIR_PATH = "data"

# Constants
DATE = datetime.utcnow().strftime("%Y_%m_%d")
CATEGORY_EXCLUDES = [
    "Concept Category",
    "Attribute Category",
    "Relationship Category",
]

# Mapping from Case IDs to observations
PC_ID_TO_PCO_DICTS: Optional[dict] = None


def main():
    global PC_ID_TO_PCO_DICTS
    print("Getting cases...")
    case_id_to_case, case_ids_to_exclude = _load_cases()

    print("Getting observations...")
    PC_ID_TO_PCO_DICTS = _load_observations()

    print("Getting solves...")
    solve_id_to_solve, solve_ids_to_exclude = _load_solves()

    print("Getting LLM responses...")
    _generate_machine_responses(case_id_to_case, solve_id_to_solve, solve_ids_to_exclude)


class User(object):
    """User object."""
    def __init__(self, user):
        # Dynamically set attributes from the user dictionary
        for key, value in user.items():
            setattr(self, key, value)

class Case(object):
    """Case object."""
    def __init__(self, case):
        # Dynamically set attributes from the case dictionary
        for key, value in case.items():
            setattr(self, key, value)

class Solve(object):
    """Solve object."""
    def __init__(self, solve):
        # Dynamically set attributes from the solve dictionary
        for key, value in solve.items():
            setattr(self, key, value)

class MachineSolve(object):
    """MachineSolve object."""
    def __init__(self, machinesolve):
        # Dynamically set attributes from the machinesolve dictionary
        for key, value in machinesolve.items():
            setattr(self, key, value)



def save_to_pickle(object_to_save, path, overwrite_existing=False):
    """
    Saves an object to a pickle file at a specified path.

    Params:
        - object_to_save: a python object to save
        - path: the path (containing filename and extension) with which to save
            the pickle file to
        - overwrite_existing (optional): default False. whether to overwrite the
            existing file (if False, the pickle file at the location will be
            appended to)
    """
    if overwrite_existing:
        try:
            os.remove(path)
        except OSError:
            pass
        f = open(path, "wb")
    else:
        f = open(path, "a+b")
    pickle.dump(object_to_save, f, pickle.HIGHEST_PROTOCOL)
    f.close()


def load_from_pickle(path, num_obj=1, pickle_options=None):
    """
    Load objects from a pickle file at a specified path. If num_obj=1, return
    the object, else return an array of objects.

    Params:
        - path: the path (containing filename and extension) to the pickle file
            with which the object will be loaded from
        - num_obj (optional): default to 1. the number of objects to load from
            the pickle file
    """
    if pickle_options is None:
        pickle_options = {}

    f = open(path, "rb")
    objs = []
    for i in range(0, num_obj):
        objs.append(pickle.load(f, **pickle_options))
    f.close()
    return objs[0] if num_obj == 1 else objs


def _load_cases():
    cases_data_path = f"{BASE_DIR_PATH}/_data_cases.pkl"
    case_id_to_case, case_ids_to_exclude = \
        load_from_pickle(path=cases_data_path, num_obj=1)
    print(f"-----> Loaded from pickle at: {cases_data_path}")
    return case_id_to_case, case_ids_to_exclude

def _load_observations():
    observations_data_path = f"{BASE_DIR_PATH}/_data_observations.pkl"
    pc_id_to_pco_dicts = load_from_pickle(path=observations_data_path, num_obj=1)
    print(f"-----> Loaded from pickle at: {observations_data_path}")
    return pc_id_to_pco_dicts


def _load_solves():
    solves_data_path = f"{BASE_DIR_PATH}/_data_solves.pkl"
    solve_id_to_solve, solve_ids_to_exclude = \
        load_from_pickle(path=solves_data_path, num_obj=1)
    print(f"-----> Loaded from pickle at: {solves_data_path}")
    return solve_id_to_solve, solve_ids_to_exclude

def _cat_helper(cats):
    cat = cats[0] if cats else None
    out = ""
    if cat:
        cat_cat = _cat_helper(cats=cat.get("categories", []))
        if cat_cat:
            out += cat_cat + " > "
        cat_name = _entity_to_str(entity_dict=cat)
        if cat_name not in CATEGORY_EXCLUDES:
            out += cat_name
    return out


def _comp_helper(comps):
    if not comps:
        return ""
    return ", ".join([_entity_to_str(entity_dict=comp) for comp in comps])


def _spec_helper(specs):
    if not specs:
        return ""
    return ", ".join([_entity_to_str(entity_dict=spec) for spec in specs])


def _rel_helper(rels):
    if not rels:
        return ""
    return ", ".join([
        _entity_to_str(
            entity_dict=rel.get("related_by")
        ) + " " + _entity_to_str(
            entity_dict=rel.get("related_entity")
        )
        for rel in rels
    ])


def _entity_to_str(entity_dict):
    # If entity has a name, use it
    name = entity_dict.get("name")
    if name:
        return name
    # Otherwise, construct it.
    negated = entity_dict.get("negated")
    cat_string = _cat_helper(cats=entity_dict.get("categories"))
    comp_string = _comp_helper(comps=entity_dict.get("components"))
    spec_string = _spec_helper(specs=entity_dict.get("specifiers"))
    rel_string = _rel_helper(rels=entity_dict.get("relations"))
    out = ""
    if cat_string:
        out += f"{cat_string}: "
    if negated:
        out += "Does not have "
    if spec_string:
        out += f"{spec_string} "
    if comp_string:
        out += comp_string
    if rel_string:
        out += f" {rel_string}"
    return out.strip()


def _case_bg_string(case):
    age = case.age
    sex = case.sex
    acuity = case.acuity
    care_setting = case.care_setting
    geography = case.geography
    chief_complaint = case.chief_complaint
    out = ""
    if age:
        out += age + " "
    if sex:
        out += sex + " "
    out += "presents"
    if acuity:
        out += " " + acuity
    if care_setting:
        out += " to the " + care_setting
    if geography:
        out += " in " + geography
    if chief_complaint:
        out += " with " + chief_complaint
    return out

def create_dict_defaultdict():
    return defaultdict(dict)

def bool_from_str(x):
    return x and isinstance(x, str) and x[0].lower() in ["y", "t"]

def get_5_most_frequent_solution_mention(case_id,
                                         case_id_to_solve_ids,
                                         solve_id_to_solve
                                         ):
    solve_ids = case_id_to_solve_ids[case_id]
    counter = Counter(
        [dia for sid in solve_ids for dia in solve_id_to_solve[sid].final_dxs])
    return [x[0] for x in counter.most_common()][:5]

def create_prompt_example_solution(case_id,
                                   case_id_to_case,
                                   solve_id_to_solve,
                                   solve_ids_to_exclude
                                   ):
    print("-----> Assigning solves to cases")
    case_id_to_solve_ids = defaultdict(set)
    case_id_to_solver_ids = defaultdict(set)
    for solve_id, solve in sorted(solve_id_to_solve.items()):
        case = case_id_to_case[solve.pc_id]
        not_creator_solve = solve.solver_id != case.creator_id
        not_resolve = solve.solver_id not in case_id_to_solver_ids[solve.pc_id]
        included_solve = solve_id not in solve_ids_to_exclude and solve.is_valid()
        if included_solve and not_creator_solve and not_resolve:
            case_id_to_solver_ids[solve.pc_id].add(solve.solver_id)
            case_id_to_solve_ids[solve.pc_id].add(solve_id)

    most_frequent_dias = get_5_most_frequent_solution_mention(case_id,
                                                              case_id_to_solve_ids,
                                                              solve_id_to_solve)
    sol_dias = case_id_to_case[case_id].diagnosis_names
    for dia in most_frequent_dias:
        if dia not in sol_dias:
            sol_dias.append(dia)
    return "\n".join(sol_dias[:5])


def _generate_machine_responses(case_id_to_case,
                                solve_id_to_solve,
                                solve_ids_to_exclude
):


    personalization = "You are a medical expert diagnosing a patient. "
    base_task = ("Provide only the most probable differential diagnosis, no explanation, "
                 "no recapitulation of the case information or task. "
                 "give a maximum of 5 answers, "
                 "sorted by probabilty of being the correct diagnosis, most probable first, "
                 "remove list numbering, and respond with each answer on a new line. "
                 "Be as concise as possible, no need to be polite."
                )

    answer_format_common = "In your answer use common shorthand non-abbreviated diagnoses. "
    answer_format_sct = "In your answer provide only the appropriate SNOMED CT fully specified name, no id. "
    self_consistency = ("Check that each differential diagnosis in your answer is "
                        "consistent with each finding in the case description. "
                        )

    few_shot_prompt = "\n\nHere are some examples of cases and their correct answers:"
    few_shot_case_ids = [3004, 4810, 2730,3775, 3704]
    for cid in few_shot_case_ids:
        case_text = _get_processed_case_text(case=case_id_to_case[cid])
        example_sol = create_prompt_example_solution(cid,
                                                     case_id_to_case,
                                                     solve_id_to_solve,
                                                     solve_ids_to_exclude
                                                     )
        few_shot_prompt += f"\n\ncase description:\n{case_text} \n\nAnswer: \n{example_sol}"

    prompt_candidates = {
        "base_common": base_task + answer_format_common,
        "base_sct": base_task + answer_format_sct,
        "base_personalization_common": personalization + base_task + answer_format_common,
        "base_personalization_sct": personalization + base_task + answer_format_sct,
        "base_personalization_common_selfconsistent": (personalization
                                                        + base_task
                                                        + answer_format_common
                                                        + self_consistency
                                                      ),
        "base_personalization_sct_selfconsistent": (personalization
                                                     + base_task
                                                     + answer_format_sct
                                                     + self_consistency
                                                    ),
        "base_personalization_sct_selfconsistent_fewshot": (personalization
                                                              + base_task
                                                              + answer_format_sct
                                                              + self_consistency
                                                              + few_shot_prompt
                                                              ),
        "base_personalization_common_selfconsistent_fewshot": (personalization
                                                                + base_task
                                                                + answer_format_common
                                                                + self_consistency
                                                                + few_shot_prompt
                                                                ),

        "base_personalization_sct_fewshot": (personalization
                                              + base_task
                                              + answer_format_sct
                                              + few_shot_prompt
                                              ),
        "base_personalization_common_fewshot": (personalization
                                                + base_task
                                                + answer_format_common
                                                + few_shot_prompt
                                                ),
        "base_sct_fewshot": ( base_task
                              + answer_format_sct
                              + few_shot_prompt
                            ),
        "base_common_fewshot": (base_task
                                + answer_format_common
                                + few_shot_prompt
                                                ),
        "base_common_selfconsistent": ( base_task
                                        + answer_format_common
                                        + self_consistency
                                                      ),
        "base_sct_selfconsistent": ( base_task
                                     + answer_format_sct
                                     + self_consistency
                                                    ),
        "base_sct_selfconsistent_fewshot": ( base_task
                                             + answer_format_sct
                                             + self_consistency
                                             + few_shot_prompt
                                                              ),
        "base_common_selfconsistent_fewshot": ( base_task
                                                + answer_format_common
                                                + self_consistency
                                                + few_shot_prompt
                                                                )
    }

    message_candidates = {
        "base_common": [
            { "role": "system", "content":  base_task },
            { "role": "user", "content": answer_format_common }
         ],
        "base_sct": [
            { "role": "system", "content":  base_task },
            { "role": "user", "content": answer_format_sct }
         ],
        "base_personalization_common": [
            { "role": "system", "content": personalization + base_task },
            { "role": "user", "content": answer_format_common }
         ],
        "base_personalization_sct": [
            { "role": "system", "content": personalization + base_task },
            { "role": "user", "content": answer_format_sct }
         ],
        "base_personalization_common_selfconsistent": [
            { "role": "system", "content": personalization + base_task },
            { "role": "user", "content": answer_format_common + self_consistency}
         ],
        "base_personalization_sct_selfconsistent": [
            { "role": "system", "content": personalization + base_task },
            { "role": "user", "content": answer_format_sct + self_consistency}
         ],
        "base_personalization_sct_selfconsistent_fewshot": [
            { "role": "system", "content": personalization + base_task },
            { "role": "user", "content": answer_format_sct + self_consistency + few_shot_prompt}
         ],
        "base_personalization_common_selfconsistent_fewshot": [
            { "role": "system", "content": personalization + base_task },
            { "role": "user", "content": answer_format_common + self_consistency + few_shot_prompt}
         ],
        "base_personalization_sct_fewshot": [
            { "role": "system", "content": personalization + base_task },
            { "role": "user", "content": answer_format_sct + few_shot_prompt}
         ],
        "base_personalization_common_fewshot": [
            { "role": "system", "content": personalization + base_task },
            { "role": "user", "content": answer_format_common + few_shot_prompt}
         ],
        "base_common_selfconsistent": [
            { "role": "system", "content": base_task },
            { "role": "user", "content": answer_format_common + self_consistency}
         ],
        "base_sct_selfconsistent": [
            { "role": "system", "content": base_task },
            { "role": "user", "content": answer_format_sct + self_consistency}
         ],
        "base_sct_selfconsistent_fewshot": [
            { "role": "system", "content": base_task },
            { "role": "user", "content": answer_format_sct + self_consistency + few_shot_prompt}
         ],
        "base_common_selfconsistent_fewshot": [
            { "role": "system", "content": base_task },
            { "role": "user", "content": answer_format_common + self_consistency + few_shot_prompt}
         ],
        "base_sct_fewshot": [
            { "role": "system", "content": base_task },
            { "role": "user", "content": answer_format_sct + few_shot_prompt}
         ],
        "base_common_fewshot": [
            { "role": "system", "content": base_task },
            { "role": "user", "content": answer_format_common + few_shot_prompt}
         ]
    }

    machine_solves_data_path = f"{BASE_DIR_PATH}/_data_machine_solves_varied_prompts_{DATE}.pkl"
    case_id_to_case_list = list(case_id_to_case.items())
    case_id_to_mkey_to_mdia = defaultdict(create_dict_defaultdict)
    model_fns = {
        "google": get_google_ai_rest_solve,
        "openai": get_open_ai_solve,
        "meta": get_meta_llama_solve,
        "anthropic": get_anthropic_ai_solve,
        "mistral": get_mistral_ai_solve,
    }
    for model_key, model_fn in model_fns.items():
        for case_id, case in case_id_to_case_list:
            case_text = _get_processed_case_text(case=case)
            for prompt_key, prompt in prompt_candidates.items():
                if "fewshot" in prompt_key:
                    prompt_end = f"\n\nWhat is the differential diagnosis for the following case description?\n\ncase description:\n{case_text} \n\nAnswer: \n"
                else:
                    prompt_end = f"\n\ncase description:\n{case_text} \n\nWhat is the differential diagnosis?"
                prompt += prompt_end
                print("---------------------------------")
                print(f"sending api call to {model_key}.")
                print(f"PROMPT: {prompt}")
                if model_key in ["openai", "meta", "anthropic", "mistral"]:
                    # use messages format for chat models
                    messages = deepcopy(message_candidates[prompt_key])
                    messages[-1]["content"] += prompt_end
                    if model_key == "meta":
                        # Llama-2 requires a nested array for the payload
                        messages = [messages]
                    model, raw_resp, final_dxs, final_dx_lists = model_fn(case_text=case_text, messages_override=messages)
                else:
                    model, raw_resp, final_dxs, final_dx_lists = model_fn(case_text=case_text, prompt_override=prompt)
                print(f"----> Solve ddx: {final_dxs}")
                print(f"----> Solve ddx lists: {final_dx_lists}")
                key = f"{model.org}_{model.name}"
                case_id_to_mkey_to_mdia[case_id][key][prompt_key] = (raw_resp, final_dxs, final_dx_lists)
                sleep(random.random()) # Do not overload the API endpoint

        save_to_pickle(
          object_to_save = case_id_to_mkey_to_mdia,
          path = machine_solves_data_path,
          overwrite_existing = True,
        )

def _get_processed_case_text(case, pco_ids=None, log=True):
    case_id = case.id
    case_bg_string = _case_bg_string(case=case) + "."
    pco_dicts = PC_ID_TO_PCO_DICTS.get(case_id)
    if pco_ids:
        pco_dicts = list(
            filter(lambda x: x.get("pco_id") in pco_ids, pco_dicts))
    if not pco_dicts:
        if log:
            print(f"----> No pco_dicts for case: {case_id}.")
        return None
    if len(list(filter(lambda x: x.get("media"), pco_dicts))) > 0:
        if log:
            print(f"----> Case {case_id} has media.")
        return None
    case_body_string = ""
    for pco_dict in pco_dicts:
        try:
            pco_string = _entity_to_str(entity_dict=pco_dict)
        except Exception as e:
            if log:
                print(f"----> Bad pco_dict: {pco_dict}. Error: {e}")
            pco_string = ""
        else:
            case_body_string += f"\n- {pco_string}"
    return f"{case_bg_string}{case_body_string}".strip()



if __name__ == "__main__":
    main()
