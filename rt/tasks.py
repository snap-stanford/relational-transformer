# tuples are (database, table, target column, leakage columns)

forecast_clf_tasks = [
    ("rel-amazon", "user-churn", "churn", []),
    ("rel-hm", "user-churn", "churn", []),
    ("rel-stack", "user-badge", "WillGetBadge", []),
    ("rel-amazon", "item-churn", "churn", []),
    ("rel-stack", "user-engagement", "contribution", []),
    ("rel-avito", "user-visits", "num_click", []),
    ("rel-avito", "user-clicks", "num_click", []),
    ("rel-event", "user-ignore", "target", []),
    ("rel-trial", "study-outcome", "outcome", []),
    ("rel-f1", "driver-dnf", "did_not_finish", []),
    ("rel-event", "user-repeat", "target", []),
    ("rel-f1", "driver-top3", "qualifying", []),
]

forecast_reg_tasks = [
    ("rel-hm", "item-sales", "sales", []),
    ("rel-amazon", "user-ltv", "ltv", []),
    ("rel-amazon", "item-ltv", "ltv", []),
    ("rel-stack", "post-votes", "popularity", []),
    ("rel-trial", "site-success", "success_rate", []),
    ("rel-trial", "study-adverse", "num_of_adverse_events", []),
    ("rel-event", "user-attendance", "target", []),
    ("rel-f1", "driver-position", "position", []),
    ("rel-avito", "ad-ctr", "num_click", []),
]

autocomplete_clf_tasks = [
    ("rel-avito", "SearchInfo", "IsUserLoggedOn", []),
    ("rel-stack", "postLinks", "LinkTypeId", []),
    ("rel-amazon", "review", "verified", []),
    ("rel-trial", "studies", "has_dmc", []),
    (
        "rel-trial",
        "eligibilities",
        "adult",
        [
            "child",
            "older_adult",
            "minimum_age",
            "maximum_age",
            "population",
            "criteria",
            "gender_description",
        ],
    ),
    (
        "rel-trial",
        "eligibilities",
        "child",
        [
            "adult",
            "older_adult",
            "minimum_age",
            "maximum_age",
            "population",
            "criteria",
            "gender_description",
        ],
    ),
    ("rel-event", "event_interest", "not_interested", ["interested"]),
]

autocomplete_reg_tasks = [
    ("rel-amazon", "review", "rating", ["review_text", "summary"]),
    (
        "rel-f1",
        "results",
        "position",
        [
            "statusId",
            "positionOrder",
            "points",
            "laps",
            "milliseconds",
            "fastestLap",
            "rank",
        ],
    ),
    ("rel-f1", "qualifying", "position", []),
    ("rel-trial", "studies", "enrollment", []),
    ("rel-f1", "constructor_results", "points", []),
    ("rel-f1", "constructor_standings", "position", ["wins", "points"]),
    ("rel-hm", "transactions", "price", []),
    ("rel-event", "users", "birthyear", []),
]

all_tasks = (
    forecast_clf_tasks
    + forecast_reg_tasks
    + autocomplete_clf_tasks
    + autocomplete_reg_tasks
)

forecast_tasks = forecast_clf_tasks + forecast_reg_tasks

all_dbs = [
    "rel-amazon",
    "rel-hm",
    "rel-stack",
    "rel-avito",
    "rel-event",
    "rel-trial",
    "rel-f1",
]
