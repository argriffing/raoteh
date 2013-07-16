
class CompoundNegLL(object):
    """
    Abbreviation of instances is cnll.
    """
    def __init__(self,
            init_prim, init_tol, dwell_prim, dwell_tol, trans_prim, trans_tol):
        self.init_prim = init_prim
        self.init_tol = init_tol
        self.dwell_prim = dwell_prim
        self.dwell_tol = dwell_tol
        self.trans_prim = trans_prim
        self.trans_tol = trans_tol

    @property
    def init(self):
        return self.init_prim + self.init_tol

    @property
    def dwell(self):
        return self.dwell_prim + self.dwell_tol

    @property
    def trans(self):
        return self.trans_prim + self.trans_tol
