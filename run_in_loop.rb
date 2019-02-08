def count_images()
  `cd ./output/ && ls *.png | wc -l`.to_i
end

start_time = Time.now
loops_to_run = 100
prev_count = count_images()
loops_to_run.times do |t|
  loop_start_time = Time.now
  puts "Running instance #{(t + 1).to_s} of #{loops_to_run}..."
  puts `./randobrot 9999 1000 ./output/`
  puts "Converting colors."
  puts `cd ./output/ && ruby convert_all.rb`
  total_count = count_images()
  puts "Generated %d new images in %f seconds" % [total_count - prev_count,
    Time.now - loop_start_time]
  puts "Generated %d images total, running for %f seconds" % [total_count,
    Time.now - start_time]
  prev_count = total_count
end
