/**
 * D3.js Force-Directed Network Visualization
 * 
 * Features:
 * - Physics simulation with adjustable forces
 * - Drag nodes to reorganize
 * - Zoom and pan
 * - Hover for details
 * - Click to highlight connections
 * - Community detection coloring
 */

class NetworkVisualizer {
    constructor(containerId, width = 1200, height = 800) {
        this.container = d3.select(`#${containerId}`);
        this.width = width;
        this.height = height;
        
        // Color scheme
        this.colors = {
            primary: '#e879f9',
            secondary: '#22d3ee',
            accent: '#0ea5e9',
            background: '#1e293b',
            text: '#f1f5f9'
        };
        
        this.svg = null;
        this.simulation = null;
        this.nodes = [];
        this.links = [];
        
        this.initSVG();
    }
    
    initSVG() {
        // Clear existing
        this.container.html('');
        
        // Create SVG
        this.svg = this.container.append('svg')
            .attr('width', this.width)
            .attr('height', this.height)
            .style('background', this.colors.background)
            .style('border-radius', '12px');
        
        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on('zoom', (event) => {
                this.svg.select('.network-group')
                    .attr('transform', event.transform);
            });
        
        this.svg.call(zoom);
        
        // Create main group
        this.svg.append('g')
            .attr('class', 'network-group');
    }
    
    renderNetwork(data) {
        /**
         * Render network from data
         * 
         * data = {
         *   nodes: [{id, label, degree, centrality}, ...],
         *   edges: [{source, target, weight}, ...]
         * }
         */
        this.nodes = data.nodes;
        this.links = data.edges;
        
        // Create force simulation
        this.simulation = d3.forceSimulation(this.nodes)
            .force('link', d3.forceLink(this.links)
                .id(d => d.id)
                .distance(d => 100 / (d.weight + 1))
            )
            .force('charge', d3.forceManyBody()
                .strength(-300)
            )
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(d => this.getNodeSize(d) + 5));
        
        const g = this.svg.select('.network-group');
        
        // Draw links
        const link = g.selectAll('.link')
            .data(this.links)
            .enter().append('line')
            .attr('class', 'link')
            .attr('stroke', this.colors.accent)
            .attr('stroke-opacity', 0.4)
            .attr('stroke-width', d => Math.sqrt(d.weight || 1) * 2);
        
        // Draw nodes
        const node = g.selectAll('.node')
            .data(this.nodes)
            .enter().append('g')
            .attr('class', 'node')
            .call(d3.drag()
                .on('start', (event, d) => this.dragstarted(event, d))
                .on('drag', (event, d) => this.dragged(event, d))
                .on('end', (event, d) => this.dragended(event, d))
            );
        
        // Node circles
        node.append('circle')
            .attr('r', d => this.getNodeSize(d))
            .attr('fill', d => this.getNodeColor(d))
            .attr('stroke', 'white')
            .attr('stroke-width', 2)
            .on('mouseover', (event, d) => this.highlightNode(event, d))
            .on('mouseout', (event, d) => this.unhighlightNode(event, d))
            .on('click', (event, d) => this.selectNode(event, d));
        
        // Node labels
        node.append('text')
            .text(d => d.label)
            .attr('dx', 12)
            .attr('dy', 4)
            .attr('fill', this.colors.text)
            .attr('font-size', '12px')
            .attr('pointer-events', 'none');
        
        // Tooltips
        node.append('title')
            .text(d => `${d.label}\nDegree: ${d.degree}\nCentrality: ${(d.centrality || 0).toFixed(3)}`);
        
        // Update positions on simulation tick
        this.simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            node.attr('transform', d => `translate(${d.x},${d.y})`);
        });
        
        // Add legend
        this.addLegend();
    }
    
    getNodeSize(d) {
        return Math.sqrt((d.centrality || 0.1) * 500) + 5;
    }
    
    getNodeColor(d) {
        const degree = d.degree || 0;
        if (degree > 10) return this.colors.primary;
        if (degree > 5) return this.colors.secondary;
        return this.colors.accent;
    }
    
    highlightNode(event, d) {
        // Highlight connected nodes and edges
        const connectedNodes = new Set([d.id]);
        this.links.forEach(link => {
            if (link.source.id === d.id) connectedNodes.add(link.target.id);
            if (link.target.id === d.id) connectedNodes.add(link.source.id);
        });
        
        this.svg.selectAll('.node circle')
            .attr('opacity', n => connectedNodes.has(n.id) ? 1 : 0.2);
        
        this.svg.selectAll('.link')
            .attr('stroke-opacity', l => 
                (l.source.id === d.id || l.target.id === d.id) ? 0.8 : 0.1
            );
    }
    
    unhighlightNode(event, d) {
        this.svg.selectAll('.node circle').attr('opacity', 1);
        this.svg.selectAll('.link').attr('stroke-opacity', 0.4);
    }
    
    selectNode(event, d) {
        console.log('Selected node:', d);
        // Could trigger additional actions
    }
    
    dragstarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    dragended(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
    
    addLegend() {
        const legend = this.svg.append('g')
            .attr('class', 'legend')
            .attr('transform', `translate(20, 20)`);
        
        const legendData = [
            {label: 'High Degree (>10)', color: this.colors.primary},
            {label: 'Medium Degree (5-10)', color: this.colors.secondary},
            {label: 'Low Degree (<5)', color: this.colors.accent}
        ];
        
        legendData.forEach((item, i) => {
            const g = legend.append('g')
                .attr('transform', `translate(0, ${i * 25})`);
            
            g.append('circle')
                .attr('r', 8)
                .attr('fill', item.color)
                .attr('stroke', 'white')
                .attr('stroke-width', 2);
            
            g.append('text')
                .attr('x', 20)
                .attr('y', 5)
                .attr('fill', this.colors.text)
                .attr('font-size', '12px')
                .text(item.label);
        });
    }
    
    exportAsSVG() {
        const svgData = this.svg.node().outerHTML;
        const blob = new Blob([svgData], {type: 'image/svg+xml;charset=utf-8'});
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'network.svg';
        link.click();
        URL.revokeObjectURL(url);
    }
}

// Global instance
let networkViz = null;

function initializeNetwork(containerId, data) {
    networkViz = new NetworkVisualizer(containerId);
    networkViz.renderNetwork(data);
    return networkViz;
}

