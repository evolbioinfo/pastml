from pastml import numeric2datetime
from pastml.tree import DATE, DATE_CI


def get_formatted_date(node, dates_are_dates):
    date, ci = getattr(node, DATE), getattr(node, DATE_CI, None)
    if dates_are_dates:
        try:
            date = numeric2datetime(date).strftime("%d %b %Y")
            if ci is not None:
                ci = [numeric2datetime(ci[0]).strftime("%d %b %Y"), numeric2datetime(ci[1]).strftime("%d %b %Y")]
        except:
            pass
    return date if ci is None else '{} ({}-{})'.format(str(date), str(ci[0]), str(ci[1]))
