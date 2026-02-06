"""
Snowflake Connection Configuration
For MLB Baseball Model POC
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class SnowflakeConfig:
    """Snowflake connection parameters"""
    
    # Connection details
    USER = os.getenv('SNOWFLAKE_USER', 'your_username')
    PASSWORD = os.getenv('SNOWFLAKE_PASSWORD', 'your_password')
    ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT', 'your_account_id')
    WAREHOUSE = os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH')
    DATABASE = os.getenv('SNOWFLAKE_DATABASE', 'BASEBALL_DB')
    SCHEMA = os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC')
    ROLE = os.getenv('SNOWFLAKE_ROLE', 'ACCOUNTADMIN')
    
    # Snowflake Stages (for storing models/artifacts)
    MODELS_STAGE = '@MODELS_STAGE'
    DATA_STAGE = '@DATA_STAGE'
    
    # Tables
    GAMES_TABLE = 'GAMES'
    MODELS_METADATA_TABLE = 'MODELS_METADATA'
    PREDICTIONS_LOG_TABLE = 'PREDICTIONS_LOG'
    
    @staticmethod
    def get_connection_params():
        """Return connection parameters for snowflake-connector-python"""
        return {
            'user': SnowflakeConfig.USER,
            'password': SnowflakeConfig.PASSWORD,
            'account': SnowflakeConfig.ACCOUNT,
            'warehouse': SnowflakeConfig.WAREHOUSE,
            'database': SnowflakeConfig.DATABASE,
            'schema': SnowflakeConfig.SCHEMA,
            'role': SnowflakeConfig.ROLE
        }


# Setup instructions
SETUP_INSTRUCTIONS = """
SNOWFLAKE MLOps POC - SETUP GUIDE
==================================

1. CREATE .env FILE in snowflake_poc/ directory:
   ------------------------------------------------
   SNOWFLAKE_USER=your_email@company.com
   SNOWFLAKE_PASSWORD=your_password
   SNOWFLAKE_ACCOUNT=xy12345
   SNOWFLAKE_WAREHOUSE=COMPUTE_WH
   SNOWFLAKE_DATABASE=BASEBALL_DB
   SNOWFLAKE_SCHEMA=PUBLIC
   SNOWFLAKE_ROLE=ACCOUNTADMIN

2. FIND YOUR ACCOUNT ID:
   ----------------------
   - Login to Snowflake
   - URL format: https://<account_id>.snowflakecomputing.com
   - Example: xy12345.us-east-1
   - Use only the <account_id> part

3. INSTALL DEPENDENCIES:
   ----------------------
   pip install snowflake-connector-python sqlalchemy pandas

4. RUN SETUP SCRIPT:
   ------------------
   python snowflake_setup.py

5. VERIFY CONNECTION:
   --------------------
   python -c "from snowflake_config import SnowflakeConfig; print(SnowflakeConfig.get_connection_params())"
"""
