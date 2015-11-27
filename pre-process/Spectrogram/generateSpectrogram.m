function [] = generateSpectrogram()
    cd ../../data/pre-process/Spectrogram/
    %Get all class directories holding the sound files
    classes = dir();
    %Go through all classes
    for i = 1:length(classes)
        if ((classes(i).isdir == 1) && (classes(i).name(1) ~= '.'))
            cd(classes(i).name);
            %Get all sound files (.wav) in class (should be only 1 or 0)
            sounds = dir('*.wav');
            for j = 1:length(sounds)
                %Generate spectrogram for wav file
                [song, fs] = audioread(sounds(j).name);
                s = spectrogram(song(:,1), 800, [], [], 'yaxis');  %not sure if need to use fs??
                %Save spectrogram as jpeg file
                splits = strsplit(sounds(j).name, '.');
                imgName = sprintf('%s_spectro.jpg', splits{1});
                imwrite(s(1:2:end,1:2:end), imgName);
                %Delete the wav file
                delete(sounds(j).name);
            end
            cd ..
        end
    end
    cd ../../../pre-process/Spectrogram/
end