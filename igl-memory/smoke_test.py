import requests

BASE = "http://127.0.0.1:8000"

USER = "bro"

# 1. M
event = requests.post(
    f"{BASE}/events",
    json={
        "user_id": USER,
        "event_type":
            "explicit_preference",

        "content": {
            "text":
                "When stress-testing my idea, "
                "propose a repair too."
        },

        "sensitivity": "private",
    },
).json()

print(
    "EVENT:",
    event["id"]
)


# 2. D
belief = requests.post(
    f"{BASE}/beliefs",
    json={
        "user_id": USER,

        "belief_type":
            "interaction_preference",

        "value": {
            "preference":
                "construct_then_critique"
        },

        "confidence": 0.90,

        "applicability": {
            "mode":
                "research"
        },

        "source_event_ids": [
            event["id"]
        ],

        "status": "ACTIVE",
    },
).json()

print(
    "BELIEF:",
    belief["id"]
)


# 3. Trust + Apply
matching = requests.post(
    f"{BASE}/gate/query",
    json={
        "user_id": USER,

        "purpose":
            "answer_chat",

        "context": {
            "mode":
                "research"
        },
    },
).json()

print(
    "MATCHING:",
    matching
)


# 4. Trusted but not applicable
wrong_context = requests.post(
    f"{BASE}/gate/query",
    json={
        "user_id": USER,

        "purpose":
            "answer_chat",

        "context": {
            "mode":
                "casual"
        },
    },
).json()

print(
    "WRONG CONTEXT:",
    wrong_context
)


# 5. User sovereignty
invalidated = requests.post(
    (
        f"{BASE}/beliefs/"
        f"{belief['id']}/invalidate"
    ),
    json={
        "action":
            "INVALIDATE",

        "source":
            "USER",

        "reason": {
            "text":
                "Do not use this preference anymore."
        },

        "confidence_delta":
            0.0,
    },
).json()

print(
    "INVALIDATED:",
    invalidated["status"]
)


# 6. Exists, but Gate refuses it
after = requests.post(
    f"{BASE}/gate/query",
    json={
        "user_id": USER,

        "purpose":
            "answer_chat",

        "context": {
            "mode":
                "research"
        },
    },
).json()

print(
    "AFTER:",
    after
)


# 7. M still exists
original = requests.get(
    f"{BASE}/events/{event['id']}"
).json()

print(
    "ORIGINAL M STILL EXISTS:",
    original["id"]
)