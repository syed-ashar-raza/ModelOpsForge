from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "modelopsforge_requests_total",
    "Total inference requests.",
    ["endpoint", "status"],
)
LATENCY = Histogram(
    "modelopsforge_request_latency_seconds",
    "Inference request latency.",
    ["endpoint"],
)
PREDICTIONS = Counter(
    "modelopsforge_predictions_total",
    "Predictions by class.",
    ["class_label"],
)
