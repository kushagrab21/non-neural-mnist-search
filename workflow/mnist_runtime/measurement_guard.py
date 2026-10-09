"""Parent-only bracketed comparison of awake performance and continuous clocks."""
import time,math
from .elapsed import continuous_seconds
from .common import Blocked

POLICY={'id':'measurement-clock-discontinuity/1','absolute_tolerance_seconds':0.001,
        'allowed_relative_drift':0.00001,'max_bracket_seconds':0.001,
        'semantics':'invalidates timing, preserves raw observations; does not diagnose sleep',
        'tolerance_basis':'1 ms conservative numerical/skew allowance plus 10 ppm allowed relative drift; bracket width independently limited to 1 ms; sub-threshold interruptions are not excluded'}


def sample():
    lower=continuous_seconds();uptime=time.monotonic();upper=continuous_seconds()
    return {'continuous_lower':lower,'performance':uptime,'continuous_upper':upper}


class MeasurementGuard:
    def __init__(self):
        self.samples=[];self.valid=True;self.failure=None
        self.observe()

    def observe(self):
        try:
            point=sample();self.samples.append(point)
            values=list(point.values())
            if any(type(x) not in (float,int) or not math.isfinite(x) for x in values):raise ValueError('nonfinite clock observation')
            width=point['continuous_upper']-point['continuous_lower']
            if width<0 or width>POLICY['max_bracket_seconds']:raise ValueError('clock read bracket outside policy')
            first=self.samples[0]
            elapsed=point['performance']-first['performance']
            if elapsed<0:raise ValueError('performance clock moved backwards')
            # Clock offset is an interval due to bracketed reads. Compare the
            # nearest ends: ordinary read skew cannot masquerade as a gap.
            initial=(first['continuous_lower']-first['performance'],first['continuous_upper']-first['performance'])
            current=(point['continuous_lower']-point['performance'],point['continuous_upper']-point['performance'])
            gap=max(0,current[0]-initial[1],initial[0]-current[1])
            if gap>POLICY['absolute_tolerance_seconds']+POLICY['allowed_relative_drift']*elapsed:
                raise ValueError('continuous/performance clocks diverged beyond policy')
        except (ValueError,OSError,Blocked) as exc:
            self.valid=False;self.failure=str(exc)
            error=Blocked('measurement_clock_discontinuity: '+str(exc));error.code='MEASUREMENT_CLOCK_DISCONTINUITY';raise error from exc
        return point

    def evidence(self):
        return {'policy':POLICY,'complete':self.valid,'samples':self.samples,'failure':self.failure}
