from stable_baselines3.common.env_checker import check_env
from quadcoptV1 import QuadcoptEnvV1

env = QuadcoptEnvV1()
# It will check your custom environment and output additional warnings if needed
check_env(env)