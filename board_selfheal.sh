#!/bin/bash
# Keep SUFFICE dashboard alive (cron-managed; curl-based liveness so zombies get replaced).
cd /home/ubuntu/wm-redundancy-paper || exit 0
pgrep -f "[v]sg_board.py" >/dev/null 2>&1 || setsid nohup python3 vsg_board.py >board.poller.log 2>&1 </dev/null &
if ! curl -s -o /dev/null --max-time 3 http://localhost:8899/ 2>/dev/null; then
  pkill -9 -f "[h]ttp.server 8899" 2>/dev/null
  ( cd board && setsid nohup python3 -m http.server 8899 --bind 0.0.0.0 >../board.http.log 2>&1 </dev/null & )
fi
