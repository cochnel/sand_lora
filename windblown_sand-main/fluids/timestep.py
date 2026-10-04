class Time:

    def __init__(self, max_dt):

        self.max_dt = max_dt
        self.dt = max_dt
        self.current_time = 0

    def set_dt(self, vel):
        self.dt = min(self.max_dt, vel.abs().max())
        return self.dt
    
    def step(self):
        self.current_time += self.dt