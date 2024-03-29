import time


class ProgressBar:
    def __init__(self, total, bar_length, update_rate=500):
        self.total = total
        self.bar_length = bar_length
        self.blinking = True
        self.last_refresh = 0
        self.update_rate = update_rate

    def update_progress_bar(self, progress):
        millis = int(round(time.time() * 1000))
        # print(str(millis)+">"+str(self.last_refresh + self.update_rate))
        if millis > (self.last_refresh + self.update_rate):
            print("", end="\r")
            percentage = progress / self.total
            bar_string = "#" * round(percentage * self.bar_length)
            if self.blinking:
                bar_string = bar_string + "."
            else:
                bar_string = bar_string + " "
            self.blinking = not self.blinking
            bar_string = bar_string + "." * round((1 - percentage) * self.bar_length)
            bar_string = bar_string + " " + str(round(percentage*100)) + "%"
            # print(bar_string)
            print(bar_string, end="")
            self.last_refresh = int(round(time.time() * 1000))
