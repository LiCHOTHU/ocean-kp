class colorPanel(object):
    def __init__(self, colorIndex):
        self.colorIndex = colorIndex

    def getColors(self):
        if self.colorIndex == 1:
            return [
                    '#1f77b4',  # muted blue
                    '#ff7f0e',  # safety orange
                    '#2ca02c',  # cooked asparagus green
                    '#d62728',  # brick red
                    '#9467bd',  # muted purple
                    '#8c564b',  # chestnut brown
                    '#e377c2',  # raspberry yogurt pink
                    '#7f7f7f',  # middle gray
                    '#bcbd22',  # curry yellow-green
                    '#17becf'  # blue-teal
                    ]
        elif self.colorIndex == 2:		
            return [
                    '#1f77b4',  # muted blue
                    '#17becf',  # blue-teal
                    '#bcbd22',  # curry yellow-green
                    '#ff7f0e',  # safety orange
                    '#2ca02c',  # cooked asparagus green
                    '#d62728',  # brick red
                    '#9467bd',  # muted purple
                    '#8c564b',  # chestnut brown
                    '#e377c2',  # raspberry yogurt pink
                    '#7f7f7f'  # middle gray
                    ]
        else:		
            return [
                    '#1f77b4',  # muted blue
                    '#2ca02c',  # cooked asparagus green
                    '#ff7f0e',  # safety orange
                    '#17becf',  # blue-teal
                    '#d62728',  # brick red
                    '#9467bd',  # muted purple
                    '#8c564b',  # chestnut brown
                    '#e377c2',  # raspberry yogurt pink
                    '#7f7f7f'  # middle gray
                    ]
