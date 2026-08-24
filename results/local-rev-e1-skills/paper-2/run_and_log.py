import sys
sys.stdout = open("follow_up_log.txt", "w")
exec(open("follow_up.py").read())
sys.stdout.close()
