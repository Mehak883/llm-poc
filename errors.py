from enum import Enum
class ErrorCodes(Enum):
    # Application level error codes
    SUCCESS = 0
    VALID_USER_ID = 100       # Indicates a valid user ID (success case)
    ALLOWED_TEXT = 101        # Indicates allowed text (success case)
    UNAUTHORIZED = 1001       # User not authenticated
    FORBIDDEN = 1002          # User lacks permission
    INVALID_TEXT = 1004       # Invalid text input
    INVALID_TYPE = 1006       # Invalid data type
    INVALID_VALUE = 1007      # Invalid value provided
    INVALID_FORMAT = 1008     # Invalid data format
    INVALID_EMAIL = 1009      # Invalid email (combines format & validity)
    INVALID_INPUT = 2001  
    INTERNAL_SERVER_ERROR = 5001    # General invalid input
    NOT_FOUND = 3001          # Resource not found   # Server-side failure           # Request timeout

def get_error_code_description(error_code):
    """
    Maps SaraErrorCodes to their description.
    """
    descriptions = {
        ErrorCodes.NOT_FOUND: "Resource not found.",
        ErrorCodes.INVALID_INPUT: "Invalid input provided.",
        ErrorCodes.UNAUTHORIZED: "Unauthorized access.",
        ErrorCodes.FORBIDDEN: "Forbidden action.",
        ErrorCodes.VALID_USER_ID: "Valid user id provided.",
        ErrorCodes.INVALID_TEXT: "Invalid text provided.",
        ErrorCodes.ALLOWED_TEXT: "Allowed text provided.",
        ErrorCodes.INVALID_TYPE: "Invalid type provided.",
        ErrorCodes.INVALID_VALUE: "Invalid value provided.",
        ErrorCodes.INVALID_FORMAT: "Invalid format provided.",
        ErrorCodes.INVALID_EMAIL: "Invalid email provided.",
        ErrorCodes.SUCCESS: "Operation successful."

    }
    # Accept both enum and int
    if isinstance(error_code, ErrorCodes):
        return descriptions.get(error_code, "Unknown error code.")
    try:
        code_enum = ErrorCodes(error_code)
        return descriptions.get(code_enum, "Unknown error code.")
    except Exception:
        return "Unknown error code."
