function [] = generateSpectrogram()
    cd ../../data/pre-process/Spectrogram/
    %Get all class directories holding the sound files
    classes = dir();
    %Go through all classes
    for i = 1:length(classes)
        if ((classes(i).isdir == 1) && (classes(i).name(1) ~= '.'))
            cd(classes(i).name);
            avg = uint8(zeros(256,768));
            %Get all sound files (.wav) in class
            sounds = dir('*.wav');
            for j = 1:length(sounds)
                %Generate spectrogram for wav file
                [song, fs] = audioread(sounds(j).name);
                s = spectrogram(song(:,1), 512, [], [], fs);
                %Save spectrogram as jpeg file
                splits = strsplit(sounds(j).name, '.');
                imgName = sprintf('%s_spectro.jpg', splits{1});
                imwrite(s, imgName);
                %Crop the spectrogram image into 256x768 dimensions
                X = imread(imgName);
                [r c] = size(s);
                start = (c-768)/2;
                stop = 767;
                X = imcrop(X, [start 0 stop 256]);
                imwrite(X, imgName);
                %Sum up the data for the average spectrogram
                avg = avg + X;
                %Delete the wav file
                delete(sounds(j).name);
            end
            %Compute and store an average spectrogram for the class
            %Stores an empty spectrogram for classes with no videos with sound
            avg = avg ./length(sounds);
            imwrite(avg, 'average_spectro.jpg');
            cd ..
        end
    end
    cd ../../../pre-process/Spectrogram/
end