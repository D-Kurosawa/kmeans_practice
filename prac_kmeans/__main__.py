"""Documents"""
from .conf import ConfigLoader
from .mypkg import exectime


@exectime.app_time
def main():
    print('Load configure file')
    conf = ConfigLoader()
    conf.load()


if __name__ == '__main__':
    main()
