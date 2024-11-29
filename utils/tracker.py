from deep_sort_realtime.deepsort_tracker import DeepSort

def initialize_tracker():
    """Initialize Deep SORT tracker."""
    return DeepSort(max_age=30, nn_budget=20)
