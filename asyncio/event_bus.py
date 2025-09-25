from collections import deque


class EventBus:
    def __init__(self):
        self.listeners = {}
        self.queue = deque()

    def subscribe(self, event_name, callback):
        self.listeners.setdefault(event_name, []).append(callback)

    def emit(self, event_name, data):
        self.queue.append((event_name, data))

    def run(self):
        while True:
            if self.queue:
                event_name, data = self.queue.popleft()
                for callback in self.listeners.get(event_name, []):
                    callback(data)


# class EventBus:
#     def __init__(self):
#         self.subscribers = {}
#
#     def subscribe(self, event_type, handler):
#         """
#         `event_type`: any hashable object
#         `handler`: a callback function for when `event_type` is published
#         """
#         self.subscribers.setdefault(event_type, []).append(handler)
#
#     def unsubscribe(self, event_type, handler):
#         """unsubscribes the the corresponding `handler` to `event_type`"""
#         if event_type in self.subscribers:
#             self.subscribers[event_type].remove(handler)
#
#     def publish(self, event_type, data=None):
#         """Trigger the event and calls all `handlers` subscribed to it"""
#         for handler in self.subscribers.get(event_type, []):
#             handler(data)
