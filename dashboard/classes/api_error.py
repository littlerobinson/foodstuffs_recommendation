class APIError(Exception):
    """Exception raised for errors in the API."""

    def __init__(self, status_code, message="API Error"):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)
