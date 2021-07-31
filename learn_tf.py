from SAC.UserInterface.UserDefinedSettings import UserDefinedSettings
from sac_tf import SoftActorCritic
from SAC.Environment.EnvironmentFactory import EnvironmentFactory


class LearningService(object):

    def run(self):
        userDefinedSettings = UserDefinedSettings()
        environmentFactory = EnvironmentFactory(userDefinedSettings)
        env = environmentFactory.generate()

        agent = SoftActorCritic(env=env, userDefinedSettings=userDefinedSettings)
        agent.run()


if __name__ == '__main__':
    learningService = LearningService()
    learningService.run()
