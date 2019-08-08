var layoutOptions = {
  name: 'preset',
  positions: function(node){
    node.position('x', node.data('node_x'));
    node.position('y', node.data('node_y'));
    return node.position();
  },
  fit: true, // whether to fit to viewport
  padding: 1, // padding on fit
};


var cy = cytoscape({
  container: document.getElementById('cy'),

  style: cytoscape.stylesheet()
    .selector('node')
      .css({
        'width': 300,
        'height': 300,
        'content': '',
        'shape': 'barrel',
        'pie-size': '100%',
        'background-color': '#909090',
        'color': '#383838',
        'text-opacity': 1,
        'text-valign': 'center',
        'text-halign': 'center',
        'font-size': 80,
        'text-halign' : 'center',
        'text-valign' : 'center',
        'min-zoomed-font-size': 12,
        'text-background-color' : '#ffffff',
        'text-background-shape' : 'roundrectangle',
        'text-background-opacity': .3,
        'text-background-padding' : 1,
      })
    .selector('node[node_name]')
      .css({
        'content': 'data(node_name)',
      })
    .selector('node[tip]')
      .css({
        'width': 500,
        'height': 500,
        'font-size': 100,
        'shape': 'ellipse',
      })
    .selector('node[unresolved]')
      .css({
        'shape': 'octagon',
        'pie-size': '90%',
        'width': 600,
        'height': 600,
        'font-size': 150,
      })
    .selector('node[fake]')
      .css({
        'shape': 'rectangle',
        'width': 1,
        'height': 1,
        'text-opacity': 0,
        'font-size': 1,
        'text-background-opacity': 0,
        'text-background-padding' : 0,
        'opacity': 0,
      })
    .selector('node[hidden>0]')
      .css({
        'opacity': 0.1,
        'width': 100,
        'height': 100,
      })
    {% for (clazz, css) in clazz2css %}
    .selector(".{{clazz}}")
        .css({
        {{css}}
        })
    {% endfor %}
    .selector('edge')
      .css({
        'width': 100,
        'font-size': 100,
        'color': 'black',
        'content': '',
        'curve-style': 'bezier',
        'target-arrow-shape': 'triangle',
        'target-arrow-color': '#909090',
        'opacity': 1,
        'text-opacity': 1,
        'line-color': '#909090',
        'text-background-color' : '#ffffff',
        'text-background-shape' : 'roundrectangle',
        'text-background-opacity': 1,
        'text-background-padding' : 4,
        'min-zoomed-font-size': 10,
      })
    .selector('edge[fake]')
      .css({
        'target-arrow-shape': 'none',
      })
    .selector('edge[edge_name]')
      .css({
        'content': 'data(edge_name)',
      })
    .selector('edge[edge_color]')
      .css({
        'target-arrow-color': 'data(edge_color)',
        'line-color': 'data(edge_color)',
        'width': 80,
      })
    .selector(':selected')
      .css({
        'background-color': 'black',
        'line-color': 'black',
        'target-arrow-color': 'black',
        'source-arrow-color': 'black',
        'pie-size': '60%',
        'opacity': 1
      })
    .selector('.faded')
      .css({
        'opacity': 0.25,
        'text-opacity': 0
      }),

  elements: {{elements}},

  layout: layoutOptions,

  ready: function(){
    window.cy = this;
  }
});

function addQtips() {
    cy.filter(function(ele, i, eles) {
        return ele.isNode() && ele.data('tooltip') !== undefined;
    } ).qtip({
        content: function(){
                return '<br><div style="overflow: auto;"><span style="white-space:nowrap;">'
                + this.data('tooltip') + '</span></div>' + '<br>id: ' + this.data('node_root_id');
            },
        show: {event: 'mouseover'},
        hide: {event: 'mouseout'},
        style: {
                classes: 'qtip-bootstrap',
        },
        position: {
            at: 'center center',
        }
    });
}

cy.minZoom(.001);
cy.maxZoom(20);
addQtips();


function toImage(){
    document.getElementById("downloader").href = cy.jpg({ full: false, quality: 1.0, scale: 2}).replace(/^data:image\/[^;]/, 'data:application/octet-stream');
}

function fit() {
    cy.fit();
}

function resetLayout() {
    cy.startBatch();
    cy.layout(layoutOptions).run();

    if (slider !== null) {
        var cur_dist_attr = 'edge_name_' + slider.value;
        var initial_dist_attr = 'edge_name_' + slider.max;
        var cur_dist, initial_dist, mile;

        if (removed !== undefined) {
            removed.forEach(function( ele ) {
                if (ele.isEdge()) {
                    ele.target().position('x', ele.target().data('node_x'));
                    ele.target().position('y', ele.target().data('node_y'));

                    // the targets of hidden nodes might need to be moved to the last visible position for LTT timeline
                    if (ele.data(initial_dist_attr) !== undefined) {
                        mile = ele.target().data('mile');
                        if (mile !== undefined) {
                            initial_dist = ele.data(initial_dist_attr);
                            cur_dist = ele.data('edge_name_' + mile);
                            ele.target().position('y', (ele.source().position('y') +
                            cur_dist * (ele.target().position('y') - ele.source().position('y')) / initial_dist));
                        }
                    }
                }
            });
        }

        cy.edges().forEach(function( ele ) {
            cur_dist = ele.data(cur_dist_attr);
            initial_dist = ele.data(initial_dist_attr);
            if (cur_dist !== undefined && cur_dist != initial_dist) {
                ele.target().position('y', (ele.source().position('y') +
                cur_dist * (ele.target().position('y') - ele.source().position('y')) / initial_dist));
            }
        });
    }
    cy.endBatch();
}

var years = {{years}};
var slider = document.getElementById("myRange");
if (slider !== null) {
    var output = document.getElementById("demo");
    output.innerHTML = years[slider.value]; // Display the default slider value

    // Update the current slider value (each time you drag the slider handle)
    var removed = cy.collection();

    slider.oninput = function() {
        output.innerHTML = years[this.value];

        cy.startBatch();

        removed.restore();
        removed = cy.remove("[mile>" + this.value + "]");

        var cur_dist_attr = 'edge_name_' + this.value;
        var initial_dist_attr = 'edge_name_' + this.max;
        var old_dist, cur_dist;

        cy.edges().forEach(function( ele ){
            old_dist = ele.data('edge_name');
            cur_dist = ele.data(cur_dist_attr);
            if (cur_dist !== undefined && cur_dist != old_dist) {
                if (ele.data(initial_dist_attr) != cur_dist) {
                    ele.target().data('hidden', 1);
                } else {
                    ele.target().data('hidden', 0);
                    console.log(ele.target().data());
                }
                ele.target().position('y', (ele.source().position('y')
                + cur_dist * (ele.target().position('y') - ele.source().position('y')) / old_dist));
                ele.data('edge_name', cur_dist);
            }
        });
        cy.endBatch();
    }
}

function toggleFullScreen() {
  var elem = document.getElementById("fullscreenDiv");
  if ((document.fullScreenElement && document.fullScreenElement !== null) ||
   (!document.mozFullScreen && !document.webkitIsFullScreen)) {
    if (elem.requestFullScreen) {
      elem.requestFullScreen();
    } else if (elem.mozRequestFullScreen) {
      elem.mozRequestFullScreen();
    } else if (elem.webkitRequestFullScreen) {
      elem.webkitRequestFullScreen(Element.ALLOW_KEYBOARD_INPUT);
    }
    document.getElementById("fsspan").className = "glyphicon glyphicon-resize-small"
    document.getElementById("fstext").innerHTML = "  Exit full screen"
  } else {
    if (document.cancelFullScreen) {
      document.cancelFullScreen();
    } else if (document.mozCancelFullScreen) {
      document.mozCancelFullScreen();
    } else if (document.webkitCancelFullScreen) {
      document.webkitCancelFullScreen();
    }
    document.getElementById("fsspan").className = "glyphicon glyphicon-fullscreen"
    document.getElementById("fstext").innerHTML = "  Full screen"
  }
}