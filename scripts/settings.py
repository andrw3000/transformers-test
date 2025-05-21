from typing import Optional

from dotenv import find_dotenv
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Read environment variables from the .env file and validate them.
    """

    my_optional_secret: Optional[str] = None

    model_config = ConfigDict(
        env_file=find_dotenv(), extra="ignore", env_file_encoding="utf-8"
    )


try:
    SETTINGS = Settings()
except Exception as err:
    print(f"Error reading settings: {err}.")
    exit(1)


if __name__ == "__main__":

    if SETTINGS.my_optional_secret:
        print("I'm going to tell you an optional secret...")
        print(f"{SETTINGS.my_optional_secret}")
    else:
        print("I don't have any optional secrets to tell you.")
        print("Please set the MY_OPTIONAL_SECRET environment variable.")
