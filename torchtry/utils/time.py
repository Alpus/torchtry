from datetime import datetime, timedelta


def time_is_up(last_time, step_minutes):
    return datetime.now() - last_time >= timedelta(minutes=step_minutes)
