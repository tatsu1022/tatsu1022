import sys

from UserInterface.LearningService import LearningService
from UserInterface.PlayAgentService import PlayAgentService


def root():

    try:
        purpose = sys.argv[1]
    except IndexError:
        print('*** choose purpose!! ***', '\n',
              '[l: learn]  or  [p: play]', '\n',
              'such as: python root.py l', '\n',
	      'such as: python root.py p ./logs/<environment name>/sac/<date>/')
        sys.exit()

    if purpose == 'l' or purpose == 'learn':
        print('learn')
        learningService = LearningService()
        learningService.run()
    elif purpose == 'p' or purpose == 'play':
        print('play')
        learned_policy_head_path = sys.argv[2]
        playAgentService = PlayAgentService()
        playAgentService.run(learned_policy_head_path)


if __name__ == '__main__':
    root()
