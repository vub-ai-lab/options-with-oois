set terminal pdf size 8cm,4cm font "Times, 11"
set output 'plot.pdf'

set xlabel 'Episode'
set ylabel 'Cumulative reward per episode'
set format x "%1.0fK"
set key bottom right maxrows 2
set style fill transparent solid 0.33 noborder

file1 = "<(python ../avg_stats.py 2 next/out*)"
file2 = "<(python ../avg_stats.py 2 nonext/out*)"
file3 = "<(python ../avg_stats.py 2 lstm_options/out*)"

plot [0:35] [4:12] \
    file3 using 1:3:4 with filledcu notitle lc rgb "#888888", file3 using 1:2 with lines title 'LSTM + Options' lc "#000000", \
    file1 using 1:3:4 with filledcu notitle lc rgb "#88ff88", file1 using 1:2 with lines title 'OOIs + Options' lc "#006600" lw 2, \
    file2 using 1:3:4 with filledcu notitle lc rgb "#ff8888", file2 using 1:2 with lines title 'Options' lc "#660000" dt 2, \
    10.0 title "Expert Policy" lc "#888888"
