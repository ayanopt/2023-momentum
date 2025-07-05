import datetime
def log(in_string : str, include_time: bool):
    with open("logs.txt", "a") as f:
        time_rn = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_str = ""
        log_str += f"{str(time_rn)}: " if include_time else "\t"
        log_str += in_string
        print(log_str, file=f)
def clear_log():
    f = open(f"logs.txt", "w")
    f.write("")
    f.close()  
    #-----------------------------------------add google drive file