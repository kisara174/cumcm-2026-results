function plot_q2(outputDir)
% Reproducible MATLAB figure from authoritative Q2 JSONs and config.
% Candidate fill is a display grid, not a feasibility or optimality proof.
base = fileparts(mfilename('fullpath'));
if nargin < 1, outputDir = fullfile(base, 'results'); end
if ~isfolder(outputDir), mkdir(outputDir); end
cfg = jsondecode(fileread(fullfile(base, '..', 'configs', 'q2.json')));
geo = jsondecode(fileread(fullfile(base, '..', 'configs', 'q1.json')));
names = {'origin', 'edge', 'oblique'};
labels = {'Central reading', 'Outward edge reading', 'Oblique edge reading'};
data = cell(1,3);
for k = 1:3
    data{k} = jsondecode(fileread(fullfile(base, 'results', ['q2-' names{k} '.json'])));
end
f = figure('Visible','off','Color','w','Position',[100 100 1450 820]);
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
for k=1:3
    r=data{k}; cells=r.prior.cells_rad_rad_m_m;
    nexttile(k); hold on; box on;
    extent=max([cells(:,4); abs(r.selected_local_m(:))*1.4]);
    xs=linspace(0,max(30,extent),55);
    ys=linspace(-max(30,extent),max(30,extent),91);
    [xx,yy]=meshgrid(xs,ys); allowed=false(size(xx));
    for n=1:numel(xx)
        q=[xx(n),abs(yy(n))]; side=sign(yy(n));
        if side==0, continue; end
        c=cells;
        if side<0, c(:,1:2)=[-cells(:,2),-cells(:,1)]; end
        allowed(n)=candidateForDisplay(q,c,cfg,geo.bearing_error_deg);
    end
    scatter(xx(allowed),yy(allowed),4,[.80 .88 .96],'filled','HandleVisibility','off');
    for j=1:size(cells,1)
        aa=linspace(cells(j,1),cells(j,2),20);
        rr=[cells(j,3)*ones(1,20),cells(j,4)*ones(1,20)];
        av=[aa fliplr(aa)];
        patch(rr.*cos(av),rr.*sin(av),[.95 .72 .26],...
            'EdgeColor','none','FaceAlpha',.7,'HandleVisibility','off');
    end
    plot(0,0,'ko','MarkerFaceColor','k','DisplayName','First point');
    q=r.selected_local_m;
    plot([0 q(1)],[0 q(2)],'-','Color',[.10 .45 .35],'LineWidth',1.5,'HandleVisibility','off');
    plot(q(1),q(2),'p','MarkerSize',12,'MarkerFaceColor',[.10 .55 .40],...
        'MarkerEdgeColor',[.10 .40 .30],'DisplayName','Adaptive second point');
    xlim([0 max(30,extent)]); ylim([-max(30,extent) max(30,extent)]);
    axis equal; grid on; xlabel('Local forward / m'); ylabel('Local lateral / m');
    title(sprintf('%s\nD = %.3f m; move + measure = %.2f s',labels{k},...
        r.audit.sampled_worst_diameter_m,r.move_and_measure_time_s));
    if k==1, legend('Location','southoutside'); end
end
old=zeros(3,1); improved=zeros(3,1); durations=zeros(3,1);
for k=1:3
    old(k)=min([data{k}.reference_candidates.sampled_worst_diameter_m]);
    improved(k)=data{k}.audit.sampled_worst_diameter_m;
    durations(k)=data{k}.move_and_measure_time_s;
end
nexttile(4); bar([old improved]); grid on;
xticklabels({'Center','Edge','Oblique'}); ylabel('Worst angular polygon diameter / m');
title('Same-prior numerical comparison'); legend({'Fixed (800, +/-600)','Adaptive'},'Location','northoutside');
nexttile(5); bar([205*ones(3,1) durations]); grid on;
xticklabels({'Center','Edge','Oblique'}); ylabel('Move + second measurement / s');
title('Movement cost'); legend({'Fixed','Adaptive'},'Location','northoutside');
nexttile(6); axis off;
text(0,.95,{'Blue dots: candidates in plotted window', 'Orange: conservative prior enclosure',...
    'Both candidate sides are evaluated.', '', 'Reception: continuous cell extrema',...
    'Selection and diameter: numerical audit', 'No global optimality certificate',...
    'Constructed cases; no simulator scores'},'Units','normalized','FontSize',11,'VerticalAlignment','top');
sgtitle('Q2: disk-aware second observation','FontSize',16);
exportgraphics(f,fullfile(outputDir,'q2-adaptive.png'),'Resolution',150);
exportgraphics(f,fullfile(outputDir,'q2-adaptive.pdf'),'ContentType','vector');
savefig(f,fullfile(outputDir,'q2-adaptive.fig'));
close(f);
end

function yes=candidateForDisplay(q,c,cfg,error)
% Same analytic cell constraints as q2_geometry; display only.
yes=false; delta=deg2rad(error); margin=cfg.geometry_margin_m;
if q(2)<=q(1)*tan(delta)+margin, return; end
if ~isempty(cfg.maximum_move_distance_m) && norm(q)>cfg.maximum_move_distance_m, return; end
phi=atan2(q(2),q(1)); rho=norm(q); lo=c(:,1); hi=c(:,2);
ends=[q(1)*cos(lo)+q(2)*sin(lo),q(1)*cos(hi)+q(2)*sin(hi)];
dmin=min(ends,[],2); dmax=max(ends,[],2);
inside=lo<=phi & phi<=hi; dmax(inside)=rho;
rr=[c(:,3),c(:,4),max(c(:,3),min(cfg.minimum_reception_radius_m,c(:,4)))];
dd=sqrt(max(0,rho^2+rr.^2-2*rr.*dmin));
slack=max(cfg.minimum_reception_radius_m,rr)-dd;
if min(slack,[],'all')<margin, return; end
closest=max(c(:,3),min(dmax,c(:,4)));
near=sqrt(max(0,rho^2+closest.^2-2*closest.*dmax));
if min(near)<=cfg.near_radius_m+margin, return; end
if q(2)<=max(max(c(:,3).*sin(hi),c(:,4).*sin(hi))), return; end
angles=[];
for j=1:size(c,1)
    for r=c(j,3:4)
        aa=c(j,1:2);
        if r<=rho
            extra=phi+[-1 1]*acos(r/rho);
            aa=[aa extra(extra>=c(j,1)&extra<=c(j,2))]; %#ok<AGROW>
        end
        angles=[angles atan2(r*sin(aa)-q(2),r*cos(aa)-q(1))]; %#ok<AGROW>
    end
end
gamma=deg2rad(cfg.minimum_measured_crossing_angle_deg);
yes=min(angles)-delta>=-pi+gamma && max(angles)+delta<=-gamma;
end
