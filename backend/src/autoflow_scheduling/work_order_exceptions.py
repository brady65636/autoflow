class WorkOrderError(Exception):
    """Base exception for work-order business operations."""


class WorkOrderNotFound(WorkOrderError):
    pass


class ConfirmationRequired(WorkOrderError):
    pass


class InvalidConfirmationToken(WorkOrderError):
    pass


class DuplicateWorkOrderRequest(WorkOrderError):
    pass


class ReservationConflict(WorkOrderError):
    pass


class CustomerOperationForbidden(WorkOrderError):
    pass
