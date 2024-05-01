function [hDist, count] = Hausdorff(gtIm, predIm)
    [gtX, gtY] = find(gtIm);
    [predX, predY] = find(predIm);
    
    count = 0;
    hDist = -1;
    for i = 1:length(predX)
        minDist = Inf;
        for j = 1:length(gtX)
            d = sqrt((gtX(j) - predX(i))^2 + (gtY(j) - predY(i))^2);
            if d < minDist
                minDist = d;
            end
            count = count + 1;
        end
        
        if hDist < minDist
            hDist = minDist;
        end
    end
end

