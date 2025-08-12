PID="$(pgrep -f main:app)"
if [[ -n "$PID" ]]
then
    PGID="$(ps --no-headers -p $PID -o pgid)"
    kill -9 -- -${PGID// /}
fi