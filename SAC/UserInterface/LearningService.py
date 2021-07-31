from .UserDefinedSettings import UserDefinedSettings
from SoftActorCritic.SACAgent import SACAgent
from Environment.EnvironmentFactory import EnvironmentFactory


class LearningService(object):

    def run(self):
        userDefinedSettings = UserDefinedSettings()
        environmentFactory = EnvironmentFactory(userDefinedSettings)
        env = environmentFactory.generate()

        agent = SACAgent(env=env, userDefinedSettings=userDefinedSettings)
        agent.run()
