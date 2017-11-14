set terminal pdf size 8cm,4cm font "Times, 11"
set output 'plot.pdf'

set xlabel 'Episode'
set ylabel 'Cumulative reward per episode'
set format x "%1.0fK"
set key at 8, 8
set style fill transparent solid 0.33 noborder

file1 = "<(python ../avg_stats.py 2 next14/out*)"
file2 = "<(python ../avg_stats.py 2 nonext14/out*)"
file3 = "<(python ../avg_stats.py 2 next8/out*)"
file4 = "<(python ../avg_stats.py 2 next4/out*)"
file5 = "<(python ../avg_stats.py 2 lstm14/out*)"

plot [0:20] [-1:9] \
    file5 using 1:3:4 with filledcu notitle lc rgb "#888888", file5 using 1:2 with lines title 'LSTM (14)' lc "#000000", \
    file1 using 1:3:4 with filledcu notitle lc rgb "#88ff88", file1 using 1:2 with lines title 'With OOIs (14)' lc "#006600" lw 2, \
    file3 using 1:3:4 with filledcu notitle lc rgb "#88ffCC", file3 using 1:2 every 2 with lines title 'With OOIs (8)' lc "#007722" lw 2 dt 4, \
    file4 using 1:3:4 with filledcu notitle lc rgb "#88ffCC", file4 using 1:2 with lines title 'With OOIs (4)' lc "#007722" lw 2 dt 2, \
    file2 using 1:3:4 with filledcu notitle lc rgb "#ff8888", file2 using 1:2 with lines title 'Without OOIs (14)' lc "#660000" dt 2, \
    8.2 title "Optimal Policy" lc "#888888"
