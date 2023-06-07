
from ..domain import Domain
from Utils.software_utils import MatchMetric


class SoftwareDefect(Domain):
    def __init__(self, args):
        # Calling constructors of Domain
        Domain.__init__(self, args)

        self.args = args

        self.Xsource, self.Ysource, self.Xtarget, self.Ytarget, self.lenSource, self.loc = MatchMetric(
            args.DATA.GLOBAL_CSV, args.DATA.TARGET_CSV, split=True, merge=True)

        print(len(self.Xsource))







