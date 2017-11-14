set terminal pdf size 8cm,4cm font "Times, 11"
set output 'plot.pdf'

set xlabel 'Episode'
set ylabel 'Cumulative reward per episode'
set format x "%1.0fK"
set key left center
set style fill transparent solid 0.33 noborder

file1 = "<(python ../avg_stats.py 2 next/out*)"
file2 = "<(python ../avg_stats.py 2 nonext/out*)"
file3 = "<(python ../avg_stats.py 2 lstm_options/out*)"
file4 = "<(python ../avg_stats.py 2 random/out-100-0.001-16-*)"

plot [0:150] [-1:22] \
    file1 using 1:3:4 with filledcu notitle lc rgb "#88ff88", file1 using 1:2 with lines title 'OOIs + Options' lc "#006600" lw 2, \
    file4 using 1:3:4 with filledcu notitle lc rgb "#88ffCC", file4 every 4 using 1:2 with lines title 'Rnd. OOIs + 16 Options' lc "#007722" lw 2 dt 2, \
    file3 using 1:3:4 with filledcu notitle lc rgb "#888888", file3 using 1:2 with lines title 'LSTM + Options' lc "#000000", \
    file2 using 1:3:4 with filledcu notitle lc rgb "#ff8888", file2 using 1:2 with lines title 'Options' lc "#660000" dt 2, \
    20 title "Optimal Policy" lc "#888888"
