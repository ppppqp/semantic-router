# convert the MMLU dataset into PreferenceTrainingExample format
from pathlib import Path
from typing import Iterable, List
from datasets import load_dataset
from dataset_pipeline_sharegpt import (
    Turn,
    ShareGPTConversation,
    RoutePolicy,
)
from preference_model_ft_reranking import (
    PreferenceTrainingExample,
)


def build_clinc150_training_examples(use_domain_only: bool = False):
    dataset = load_dataset("contemmcm/clinc150", split="complete")
    examples: List[PreferenceTrainingExample] = []
    domain_map = {
        "0": "none_of_the_above",
        "1": "banking",
        "2": "credit_cards",
        "3": "kitchen_and_dining",
        "4": "home",
        "5": "auto_and_commute",
        "6": "travel",
        "7": "utility",
        "8": "work",
        "9": "small_talk",
        "10": "meta",
    }
    label_map = {
        "0": "oos:none_of_the_above",
        "1": "banking:freeze_account",
        "2": "banking:routing",
        "3": "banking:pin_change",
        "4": "banking:bill_due",
        "5": "banking:pay_bill",
        "6": "banking:account_blocked",
        "7": "banking:interest_rate",
        "8": "banking:min_payment",
        "9": "banking:bill_balance",
        "10": "banking:transfer",
        "11": "banking:order_checks",
        "12": "banking:balance",
        "13": "banking:spending_history",
        "14": "banking:transactions",
        "15": "banking:report_fraud",
        "16": "credit_cards:replacement_card_duration",
        "17": "credit_cards:expiration_date",
        "18": "credit_cards:damaged_card",
        "19": "credit_cards:improve_credit_score",
        "20": "credit_cards:report_lost_card",
        "21": "credit_cards:card_declined",
        "22": "credit_cards:credit_limit_change",
        "23": "credit_cards:apr",
        "24": "credit_cards:redeem_rewards",
        "25": "credit_cards:credit_limit",
        "26": "credit_cards:rewards_balance",
        "27": "credit_cards:application_status",
        "28": "credit_cards:credit_score",
        "29": "credit_cards:new_card",
        "30": "credit_cards:international_fees",
        "31": "kitchen_and_dining:food_last",
        "32": "kitchen_and_dining:confirm_reservation",
        "33": "kitchen_and_dining:how_busy",
        "34": "kitchen_and_dining:ingredients_list",
        "35": "kitchen_and_dining:calories",
        "36": "kitchen_and_dining:nutrition_info",
        "37": "kitchen_and_dining:recipe",
        "38": "kitchen_and_dining:restaurant_reviews",
        "39": "kitchen_and_dining:restaurant_reservation",
        "40": "kitchen_and_dining:meal_suggestion",
        "41": "kitchen_and_dining:restaurant_suggestion",
        "42": "kitchen_and_dining:cancel_reservation",
        "43": "kitchen_and_dining:ingredient_substitution",
        "44": "kitchen_and_dining:cook_time",
        "45": "kitchen_and_dining:accept_reservations",
        "46": "home:what_song",
        "47": "home:play_music",
        "48": "home:todo_list_update",
        "49": "home:reminder",
        "50": "home:reminder_update",
        "51": "home:calendar_update",
        "52": "home:order_status",
        "53": "home:update_playlist",
        "54": "home:shopping_list",
        "55": "home:calendar",
        "56": "home:next_song",
        "57": "home:order",
        "58": "home:todo_list",
        "59": "home:shopping_list_update",
        "60": "home:smart_home",
        "61": "auto_and_commute:current_location",
        "62": "auto_and_commute:oil_change_when",
        "63": "auto_and_commute:oil_change_how",
        "64": "auto_and_commute:uber",
        "65": "auto_and_commute:traffic",
        "66": "auto_and_commute:tire_pressure",
        "67": "auto_and_commute:schedule_maintenance",
        "68": "auto_and_commute:gas",
        "69": "auto_and_commute:mpg",
        "70": "auto_and_commute:distance",
        "71": "auto_and_commute:directions",
        "72": "auto_and_commute:last_maintenance",
        "73": "auto_and_commute:gas_type",
        "74": "auto_and_commute:tire_change",
        "75": "auto_and_commute:jump_start",
        "76": "travel:plug_type",
        "77": "travel:travel_notification",
        "78": "travel:translate",
        "79": "travel:flight_status",
        "80": "travel:international_visa",
        "81": "travel:timezone",
        "82": "travel:exchange_rate",
        "83": "travel:travel_suggestion",
        "84": "travel:travel_alert",
        "85": "travel:vaccines",
        "86": "travel:lost_luggage",
        "87": "travel:book_flight",
        "88": "travel:book_hotel",
        "89": "travel:carry_on",
        "90": "travel:car_rental",
        "91": "utility:weather",
        "92": "utility:alarm",
        "93": "utility:date",
        "94": "utility:find_phone",
        "95": "utility:share_location",
        "96": "utility:timer",
        "97": "utility:make_call",
        "98": "utility:calculator",
        "99": "utility:definition",
        "100": "utility:measurement_conversion",
        "101": "utility:flip_coin",
        "102": "utility:spelling",
        "103": "utility:time",
        "104": "utility:roll_dice",
        "105": "utility:text",
        "106": "work:pto_request_status",
        "107": "work:next_holiday",
        "108": "work:insurance_change",
        "109": "work:insurance",
        "110": "work:meeting_schedule",
        "111": "work:payday",
        "112": "work:taxes",
        "113": "work:income",
        "114": "work:rollover_401k",
        "115": "work:pto_balance",
        "116": "work:pto_request",
        "117": "work:w2",
        "118": "work:schedule_meeting",
        "119": "work:direct_deposit",
        "120": "work:pto_used",
        "121": "small_talk:who_made_you",
        "122": "small_talk:meaning_of_life",
        "123": "small_talk:who_do_you_work_for",
        "124": "small_talk:do_you_have_pets",
        "125": "small_talk:what_are_your_hobbies",
        "126": "small_talk:fun_fact",
        "127": "small_talk:what_is_your_name",
        "128": "small_talk:where_are_you_from",
        "129": "small_talk:goodbye",
        "130": "small_talk:thank_you",
        "131": "small_talk:greeting",
        "132": "small_talk:tell_joke",
        "133": "small_talk:are_you_a_bot",
        "134": "small_talk:how_old_are_you",
        "135": "small_talk:what_can_i_ask_you",
        "136": "meta:change_speed",
        "137": "meta:user_name",
        "138": "meta:whisper_mode",
        "139": "meta:yes",
        "140": "meta:change_volume",
        "141": "meta:no",
        "142": "meta:change_language",
        "143": "meta:repeat",
        "144": "meta:change_accent",
        "145": "meta:cancel",
        "146": "meta:sync_device",
        "147": "meta:change_user_name",
        "148": "meta:change_ai_name",
        "149": "meta:reset_settings",
        "150": "meta:maybe",
    }
    label_to_description = {
        "none_of_the_above": "out of scope",
        "banking": "banking related inquiries",
        "credit_cards": "credit card related inquiries",
        "kitchen_and_dining": "kitchen and dining related inquiries, like recipes or cooking",
        "home": "home related inquiries, like household tasks or furniture",
        "auto_and_commute": "automobile and commute related inquiries",
        "travel": "travel related inquiries, like booking or destinations",
        "utility": "utility related inquiries, like performing a simple task",
        "work": "work related inquiries, like office tasks or professional communication",
        "small_talk": "small talk, like casual conversation or jokes",
        "meta": "meta inquiries",
    }
    if use_domain_only:
        all_labels = list(domain_map.values())
    else:
        all_labels = list(label_map.values())
    print(all_labels)
    for index, item in enumerate(dataset):
        question = item["text"]
        # we only care about domain for routing purpose
        label = (
            domain_map[str(item["domain"])]
            if use_domain_only
            else label_map[str(item["intent"])]
        )
        sample_id = str(index)

        conversation = ShareGPTConversation(
            sample_id=sample_id,
            messages=[
                Turn(role="user", content=question),
            ],
        )

        truth_policy = RoutePolicy(
            label=label,
            description=label_to_description.get(label, label),
        )

        negative_policies = [
            RoutePolicy(label=l, description=label_to_description.get(l, l))
            for l in all_labels[:16]
            if l != label
        ]

        examples.append(
            PreferenceTrainingExample(
                conversation=conversation,
                truth_policy=truth_policy,
                negative_policies=negative_policies,
            )
        )
    return examples


def build_mmlu_training_examples():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    examples: List[PreferenceTrainingExample] = []

    all_categories = dataset.unique("category")
    for item in dataset:
        question = item["question"]
        # we only care about category for routing purpose
        category = item["category"]
        sample_id = item["question_id"]

        conversation = ShareGPTConversation(
            sample_id=sample_id,
            messages=[
                Turn(role="user", content=question),
            ],
        )

        truth_policy = RoutePolicy(
            label=category,
            description=category,
        )

        negative_policies = [
            RoutePolicy(label=cat, description=cat)
            for cat in all_categories
            if cat != category
        ]

        examples.append(
            PreferenceTrainingExample(
                conversation=conversation,
                truth_policy=truth_policy,
                negative_policies=negative_policies,
            )
        )
    return examples


if __name__ == "__main__":
    build_mmlu_training_examples()
