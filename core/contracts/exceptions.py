from http import HTTPStatus


class FileSizeExceededException(Exception):
    pass


class RemoteResourceAuthenticationException(Exception):
    def __init__(self, message=None, http_status_code=None):
        super().__init__(message)
        self.http_status_code = http_status_code

    @staticmethod
    def is_http_authentication_failure(http_status_code: HTTPStatus):
        return http_status_code in [
            HTTPStatus.UNAUTHORIZED,
            HTTPStatus.FORBIDDEN
        ]
