import sys
sys.stdout = open("follow_up2_log.txt", "w")
exec(open("follow_up2.py").read())
sys.stdout.close()
