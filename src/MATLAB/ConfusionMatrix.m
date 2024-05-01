function [confMat, tpr, fpr, dice] = ConfusionMatrix(gtMask, predMask, showTable)
    gtIm = gtMask;
    predIm = predMask;
    if isstruct(gtMask)
        gtIm = gtMask.mask;
        predIm = predMask.mask;
    end
    
    total = numel(gtIm);
    
    gtP = sum(gtIm(:));
    gtN = total - gtP;
    predP = sum(predIm(:));
    predN = total - gtN;
    
    tp = sum(sum(sum(and(predIm, gtIm))));
    tn = sum(sum(sum(and(~predIm, ~gtIm))));
    fp = sum(sum(sum(and(predIm, ~gtIm))));
    fn = sum(sum(sum(and(~predIm, gtIm))));
    
    tpr = tp / (tp + fn);
    fpr = fp / (fp + tn);
    dice = 2 * tp / (2*tp + fn + fp);
    
    confMat = 0;
    if ~showTable
        return
    end
    
    tbl = array2table(zeros(3,3),...
        'RowNames', {'predPos', 'predNeg', 'gtTotal'},...
        'VariableNames', {'gtPos', 'gtNeg', 'predTotal'});
    
    test = tp + tn + fp + fn;
    
    tbl{1,1} = tp;
    tbl{1,2} = fp;
    tbl{2,1} = fn;
    tbl{2,2} = tn;
    tbl{3,1} = gtP;
    tbl{3,2} = gtN;
    tbl{1,3} = predP;
    tbl{2,3} = predN;
    tbl{3,3} = total;   
    
    confMat = table2array(tbl(1:2, 1:2));
    
    fprintf('Total = %d, Test = %d\nTotal == Test? %d\n', total, test, total==test);
    disp(tbl);

    figure;
    imagesc(confMat);
    axis image;
    colormap parula
    colorbar;
    
    text('String','True Positive','Units','normalized','HorizontalAlignment','center','Position',[0.25, 0.95]);
    text('String','False Positive','Units','normalized','HorizontalAlignment','center','Position',[0.75, 0.95]);
    text('String','False Negative','Units','normalized','HorizontalAlignment','center','Position',[0.25, 0.45]);
    text('String','True Negative','Units','normalized','HorizontalAlignment','center','Position',[0.75, 0.45]);

    text('String',num2str(tp),'Units','normalized','HorizontalAlignment','center','Position',[0.25, 0.75]);
    text('String',num2str(fp),'Units','normalized','HorizontalAlignment','center','Position',[0.75, 0.75]);
    text('String',num2str(fn),'Units','normalized','HorizontalAlignment','center','Position',[0.25, 0.25]);
    text('String',num2str(tn),'Units','normalized','HorizontalAlignment','center','Position',[0.75, 0.25]);
end