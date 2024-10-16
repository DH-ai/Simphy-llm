//https://www.html5canvastutorials.com/tutorials/html5-canvas-bursting-particle-effect/
var canvas = Widgets.getCanvas("canvas").getContext();
var canvas_score = Widgets.getCanvas("canvasScore").getContext();
var btn_pause = Widgets.getButton("btn_pause");
var nextblock = "";
var nextblockImg = new Image();
canvas.setStrokeStyle("black");
canvas.setFillStyle("white");
function LinearMotion(a) {
  this._duration = a || 1000;
  this._date = new Date() * 1;
  this._path = [0, 0];
  this._target = 0;
}
LinearMotion.prototype = {
  reset: function () {
    this._path = [0, 0];
    this._target = 0;
  },
  setTarget: function (b) {
    var c = this.getPosition();
    var a = new Date() - this._date;
    this._date += a;
    a /= this._duration;
    this._target = b;
    b -= c;
    this._path = [b, c];
  },
  getPosition: function () {
    var a = (new Date() - this._date) / this._duration;
    if (a < 0 || a >= 1) {
      return this._target;
    }
    return this._path[0] * a + this._path[1];
  },
};

function CubicMotion(a) {
  this._duration = a || 1000;
  this._date = new Date() * 1;
  this._path = [0, 0, 0, 0];
  this._target = 0;
}
CubicMotion.prototype = {
  reset: function () {
    this._path = [0, 0, 0, 0];
    this._target = 0;
  },
  setTarget: function (b) {
    var d = this.getPosition();
    var a = new Date() - this._date;
    var c = 0;
    this._date += a;
    a /= this._duration;
    if (a >= 0 && a < 1) {
      c = (3 * this._path[0] * a + 2 * this._path[1]) * a + this._path[2];
    }
    this._target = b;
    b -= d;
    this._path = [c - 2 * b, 3 * b - 2 * c, c, d];
  },
  getPosition: function () {
    var a = (new Date() - this._date) / this._duration;
    if (a < 0 || a >= 1) {
      return this._target;
    }
    return (
      ((this._path[0] * a + this._path[1]) * a + this._path[2]) * a +
      this._path[3]
    );
  },
};

function Motion(a) {
  this._duration = a || 1000;
  this._date = new Date() * 1;
  this._path = [0, 0, 0, 0, 0, 0];
  this._target = 0;
}
Motion.prototype = {
  reset: function () {
    this._path = [0, 0, 0, 0, 0, 0];
    this._target = 0;
  },
  setTarget: function (b) {
    var e = this.getPosition();
    var a = new Date() - this._date;
    var d = 0;
    var c = 0;
    this._date += a;
    a /= this._duration;
    if (a >= 0 && a < 1) {
      c =
        (((5 * this._path[0] * a + 4 * this._path[1]) * a + 3 * this._path[2]) *
          a +
          2 * this._path[3]) *
          a +
        this._path[4];
      d =
        ((20 * this._path[0] * a + 12 * this._path[1]) * a +
          6 * this._path[2]) *
          a +
        2 * this._path[3];
    }
    this._target = b;
    b -= e;
    this._path = [
      -d / 2 - c * 3 + b * 6,
      (d * 3) / 2 + c * 8 - b * 15,
      (-d * 3) / 2 - c * 6 + b * 10,
      d / 2,
      c,
      e,
    ];
  },
  getPosition: function () {
    var a = (new Date() - this._date) / this._duration;
    if (a < 0 || a >= 1) {
      return this._target;
    }
    return (
      ((((this._path[0] * a + this._path[1]) * a + this._path[2]) * a +
        this._path[3]) *
        a +
        this._path[4]) *
        a +
      this._path[5]
    );
  },
};
var MENU_QUOTES = [
  '"If you love someone, put their name in a <B>circle</B>; because hearts can be broken, but <B>circles</B> never end."<BR><SPAN>- Brian Littrell</SPAN>',
  '"I made a <B>circle</B> with a smile for a mouth on yellow paper, because it was sunshiny and bright."<BR><SPAN>- Harvey Ball</SPAN>',
  '"A <B>circle</B> may be small, yet it may be as mathematically beautiful and perfect as a large one."<BR><SPAN>- Isaac Disraeli</SPAN>',
  '"When the tribe first sat down in a <B>circle</B> and agreed to allow only one person to speak at a time - that was the longest step forward in the history of law"<BR><SPAN>- Judge Curtis Bok</SPAN>',
  '"The nature of God is a <B>circle</B> of which the center is everywhere and the circumference is nowhere"<BR><SPAN>- Empedocles</SPAN>',
  '"The mind petrifies if a <B>circle</B> be drawn around it, and it can hardly be that dogma draws a <B>circle</B> round the mind."<BR><SPAN>- George Moore</SPAN>',
  "\"Let mathematicians and geometrician 'talk of <B>circles</B>' and triangles' charms, The figure I prize is a girl with bright eyes, And the <B>circle</B> that's formed by her arms\"<BR><SPAN>- Anonymous</SPAN>",
  '"Round, like a <B>circle</B> in a spiral<BR>Like a wheel within a wheel."<BR><SPAN>- Sting</SPAN>',
];
var UI = new (function () {
  var c = true;

  function d(m, k, h, f) {
    c = false;
    var e;
    var l = 0;
    var o;
    if (h == "x") {
      h = "scrollLeft";
    }
    if (h == "y") {
      h = "scrollTop";
    }
    var n = App.setInterval(function () {
      l++;
      o = (1 - Math.cos((l * Math.PI) / 20)) / 2;
      o = l == 20 ? k : m * (1 - o) + k * o;
      //g("menu_area")[h] = o >> 0;
      if (l == 20) {
        c = true;
        clearInterval(n);
        if (f) {
          f();
        }
      }
    }, 30);
  }
  function b() {
    //canvas.setSize(460, 450);
    /*
g("menu").style.display = "block";
g("menu_area").scrollLeft = 321;
g("menu_area").scrollTop = 157;

g("gameover").style.display = "none";
g("close").style.top = "264px";
g("close").style.left = "425px";
*/
    c = true;
    Game.mode = 1;
    Game.clear();
    Game.drawCylinder();
    renderScore();
  }
  this.init = function () {
    b();
  };
  this.setGameMode = function (h) {
    c = false;
    /*
g("menu").style.display = "none";
g("gameover").style.display = "none";
window.widget && window.resizeTo(380, 450);
for (var f = 1; f <= 3; f++) {
    g("title" + f).style.display = (f == h ? "block" : "none")
}
g("promo").innerHTML = PROMOS[Math.random() * PROMOS.length >> 0];
g("promo").style.display = g("panel").style.display = "block";
*/
    var e, k;
    if (h == 1) {
      (e = 328), (k = 202);
    }
    if (h == 2) {
      (e = 328), (k = 197);
    }
    if (h == 3) {
      (e = 330), (k = 170);
    }
    //g("close").style.left = e + "px";
    //g("close").style.top = k + "px"
  };
  this.pauseGame = function () {
    a.start(true);
  };
  this.resumeGame = function () {
    a.start(false);
  };
  var a = {
    motion: new CubicMotion(300),
    interval: null,
    frame: function () {
      var e = a;
      var f = e.motion.getPosition();
      /*
    g("canvas").style.opacity = 1 - f;
    g("paused").style.opacity = f;
    g("paused").style.display = f ? "block" : "none";
    */
      if (f == e.motion._target) {
        clearInterval(e.interval);
        e.interval = null;
      }
    },
    start: function (e) {
      this.motion.setTarget(e ? 1 : 0);
      if (!this.interval) {
        // this.interval = App.setInterval(this.frame, 20);
      }
    },
  };
  this.gameOver = function () {
    /**
g("promo").style.display = g("panel").style.display = g("paused").style.display = "none";
g("close").style.left = "435px";
g("close").style.top = "185px";
window.widget && window.resizeTo(470, 450)
*/
  };
})();

var Game = new (function () {
  var A = this;
  var E = null;
  var D = null;
  var m = 0;
  var k;
  var p;
  var v = -2;
  var n;
  var d;
  var e;
  this.displayGold = 0;
  this.innerRadius = 40;
  this.mode = 0;
  this.lines = 0;
  this.score = 0;
  this.time = 0;
  this.paused = true;
  this.started = false;
  var b;
  var a = new CubicMotion(250);

  function H(i, O, N, M) {
    N = N ? Game.innerRadius : 60;
    var L = 30;
    return {
      x: (N * Math.cos(2 * Math.PI * ((i - M) / 15 - 1 / 4)) * (O + L)) / L,
      y:
        200 -
        (O * 20 * (O / 2 + 60)) / 60 -
        0.3 * N * Math.sin(2 * Math.PI * ((i - M) / 15 - 1 / 4)),
    };
  }
  this.drawFront = function (S, R, i, L, N, M, P, O) {
    canvas.beginPath();
    var Q = [
      H(S - 0.015, R + 0.015, i, L),
      H(S + 1.015, R + 0.015, i, L),
      H(S + 1.015, R - 1.015, i, L),
      H(S - 0.015, R - 1.015, i, L),
    ];
    canvas.moveTo(Q[0].x, Q[0].y);
    canvas.lineTo(Q[1].x, Q[1].y);
    canvas.lineTo(Q[2].x, Q[2].y);
    canvas.lineTo(Q[3].x, Q[3].y);
    canvas.closePath();
    canvas.fill();
    if (P || O || N || M) {
      canvas.beginPath();
      if (P) {
        canvas.moveTo(Q[1].x, Q[1].y);
      } else {
        canvas.moveTo(Q[0].x, Q[0].y);
        canvas.lineTo(Q[1].x, Q[1].y);
      }
      if (M) {
        canvas.moveTo(Q[2].x, Q[2].y);
      } else {
        canvas.lineTo(Q[2].x, Q[2].y);
      }
      if (O) {
        canvas.moveTo(Q[3].x, Q[3].y);
      } else {
        canvas.lineTo(Q[3].x, Q[3].y);
      }
      if (!N) {
        canvas.lineTo(Q[0].x, Q[0].y);
      }
    }
    canvas.stroke();
  };
  this.drawTop = function (L, Q, M, i, N, P) {
    var O = [
      H(L - 0.015, Q, false, M),
      H(L - 0.015, Q, true, M),
      H(L + 1.015, Q, true, M),
      H(L + 1.015, Q, false, M),
    ];
    canvas.beginPath();
    canvas.moveTo(O[0].x, O[0].y);
    canvas.lineTo(O[1].x, O[1].y);
    canvas.lineTo(O[2].x, O[2].y);
    canvas.lineTo(O[3].x, O[3].y);
    canvas.closePath();
    canvas.fill();
    if (P) {
      return;
    }
    if (i || N) {
      canvas.beginPath();
      if (i) {
        canvas.moveTo(O[1].x, O[1].y);
      } else {
        canvas.moveTo(O[0].x, O[0].y);
        canvas.lineTo(O[1].x, O[1].y);
      }
      canvas.lineTo(O[2].x, O[2].y);
      if (N) {
        canvas.moveTo(O[3].x, O[3].y);
      } else {
        canvas.lineTo(O[3].x, O[3].y);
      }
      canvas.lineTo(O[0].x, O[0].y);
    }
    canvas.stroke();
  };
  this.drawSide = function (L, P, M, i, N) {
    var O = [
      H(L, P - 1.015, false, M),
      H(L, P + 0.015, false, M),
      H(L, P + 0.015, true, M),
      H(L, P - 1.015, true, M),
    ];
    canvas.beginPath();
    canvas.moveTo(O[0].x, O[0].y);
    canvas.lineTo(O[1].x, O[1].y);
    canvas.lineTo(O[2].x, O[2].y);
    canvas.lineTo(O[3].x, O[3].y);
    canvas.closePath();
    canvas.fill();
    if (i || N) {
      canvas.beginPath();
      canvas.moveTo(O[0].x, O[0].y);
      canvas.lineTo(O[1].x, O[1].y);
      if (i) {
        canvas.moveTo(O[2].x, O[2].y);
      } else {
        canvas.lineTo(O[2].x, O[2].y);
      }
      canvas.lineTo(O[3].x, O[3].y);
      if (!N) {
        canvas.lineTo(O[0].x, O[0].y);
      }
    }
    canvas.stroke();
  };
  var F = [
    {
      view: 0.08,
      theme: 0,
      frequency: 1,
      coords: [
        [
          {
            x: 1,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
          {
            x: 1,
            y: 1,
          },
          {
            x: 2,
            y: 1,
          },
        ],
      ],
    },
    {
      view: 0.25,
      theme: 1,
      frequency: 1,
      coords: [
        [
          {
            x: 0,
            y: 0,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
          {
            x: 3,
            y: 0,
          },
        ],
        [
          {
            x: 1,
            y: -1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 1,
            y: 1,
          },
          {
            x: 1,
            y: 2,
          },
        ],
      ],
    },
    {
      view: 0.5,
      theme: 2,
      frequency: 1,
      coords: [
        [
          {
            x: 1,
            y: 1,
          },
          {
            x: 0,
            y: 0,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
        ],
        [
          {
            x: 1,
            y: 1,
          },
          {
            x: 2,
            y: 0,
          },
          {
            x: 1,
            y: -1,
          },
          {
            x: 1,
            y: 0,
          },
        ],
        [
          {
            x: 2,
            y: 0,
          },
          {
            x: 1,
            y: -1,
          },
          {
            x: 0,
            y: 0,
          },
          {
            x: 1,
            y: 0,
          },
        ],
        [
          {
            x: 1,
            y: 1,
          },
          {
            x: 0,
            y: 0,
          },
          {
            x: 1,
            y: -1,
          },
          {
            x: 1,
            y: 0,
          },
        ],
      ],
    },
    {
      view: 0.5,
      theme: 3,
      frequency: 1,
      coords: [
        [
          {
            x: 0,
            y: 0,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
          {
            x: 0,
            y: 1,
          },
        ],
        [
          {
            x: 1,
            y: -1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 1,
            y: 1,
          },
          {
            x: 2,
            y: 1,
          },
        ],
        [
          {
            x: 0,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 2,
            y: -1,
          },
        ],
        [
          {
            x: 1,
            y: -1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 1,
            y: 1,
          },
          {
            x: 0,
            y: -1,
          },
        ],
      ],
    },
    {
      view: 0.5,
      theme: 4,
      frequency: 1,
      coords: [
        [
          {
            x: 2,
            y: 0,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 0,
            y: 0,
          },
          {
            x: 2,
            y: 1,
          },
        ],
        [
          {
            x: 1,
            y: -1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 1,
            y: 1,
          },
          {
            x: 2,
            y: -1,
          },
        ],
        [
          {
            x: 0,
            y: 0,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
          {
            x: 0,
            y: -1,
          },
        ],
        [
          {
            x: 1,
            y: -1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 1,
            y: 1,
          },
          {
            x: 0,
            y: 1,
          },
        ],
      ],
    },
    {
      view: 0.5,
      theme: 5,
      frequency: 1,
      coords: [
        [
          {
            x: 2,
            y: 1,
          },
          {
            x: 1,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 0,
            y: 0,
          },
        ],
        [
          {
            x: 1,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 0,
            y: 2,
          },
          {
            x: 0,
            y: 1,
          },
        ],
      ],
    },
    {
      view: 0.5,
      theme: 6,
      frequency: 1,
      coords: [
        [
          {
            x: 1,
            y: 1,
          },
          {
            x: 0,
            y: 1,
          },
          {
            x: 2,
            y: 0,
          },
          {
            x: 1,
            y: 0,
          },
        ],
        [
          {
            x: 1,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 2,
            y: 2,
          },
          {
            x: 2,
            y: 1,
          },
        ],
      ],
    },
    {
      view: -0.15,
      theme: 0,
      frequency: 0.025,
      coords: [
        [
          {
            x: 1,
            y: 1,
          },
          {
            x: 2,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
          {
            x: 3,
            y: 0,
          },
        ],
        [
          {
            x: 1,
            y: 1,
          },
          {
            x: 2,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
          {
            x: 1,
            y: -1,
          },
        ],
        [
          {
            x: 1,
            y: 1,
          },
          {
            x: 2,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
          {
            x: 0,
            y: 1,
          },
        ],
        [
          {
            x: 2,
            y: 2,
          },
          {
            x: 1,
            y: 1,
          },
          {
            x: 2,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
        ],
      ],
    },
    {
      view: 0.15,
      theme: 0,
      frequency: 0.025,
      coords: [
        [
          {
            x: 1,
            y: 1,
          },
          {
            x: 2,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
          {
            x: 0,
            y: 0,
          },
        ],
        [
          {
            x: 1,
            y: 2,
          },
          {
            x: 1,
            y: 1,
          },
          {
            x: 2,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
        ],
        [
          {
            x: 1,
            y: 1,
          },
          {
            x: 3,
            y: 1,
          },
          {
            x: 2,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
        ],
        [
          {
            x: 1,
            y: 1,
          },
          {
            x: 2,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
          {
            x: 2,
            y: -1,
          },
        ],
      ],
    },
    {
      view: 0.5,
      theme: 1,
      frequency: 0.025,
      coords: [
        [
          {
            x: -1,
            y: 0,
          },
          {
            x: 0,
            y: 0,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
          {
            x: 3,
            y: 0,
          },
        ],
        [
          {
            x: 1,
            y: -2,
          },
          {
            x: 1,
            y: -1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 1,
            y: 1,
          },
          {
            x: 1,
            y: 2,
          },
        ],
      ],
    },
    {
      view: 0.5,
      theme: 1,
      frequency: 0.025,
      coords: [
        [
          {
            x: 0,
            y: 0,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
        ],
        [
          {
            x: 1,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 1,
            y: -1,
          },
        ],
      ],
    },
    {
      view: 0.5,
      theme: 2,
      frequency: 0.05,
      coords: [
        [
          {
            x: 1,
            y: 2,
          },
          {
            x: 0,
            y: 1,
          },
          {
            x: 1,
            y: 1,
          },
          {
            x: 2,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
        ],
      ],
    },
    {
      view: 0.25,
      theme: 3,
      frequency: 0.05,
      coords: [
        [
          {
            x: 1,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
        ],
        [
          {
            x: 1,
            y: 1,
          },
          {
            x: 2,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
        ],
        [
          {
            x: 1,
            y: 1,
          },
          {
            x: 2,
            y: 1,
          },
          {
            x: 2,
            y: 0,
          },
        ],
        [
          {
            x: 2,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
        ],
      ],
    },
    {
      view: 0.25,
      theme: 4,
      frequency: 0.05,
      coords: [
        [
          {
            x: 2,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
        ],
        [
          {
            x: 1,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
        ],
        [
          {
            x: 1,
            y: 1,
          },
          {
            x: 2,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
        ],
        [
          {
            x: 1,
            y: 1,
          },
          {
            x: 2,
            y: 1,
          },
          {
            x: 2,
            y: 0,
          },
        ],
      ],
    },
    {
      view: 0.5,
      theme: 5,
      frequency: 0.025,
      coords: [
        [
          {
            x: 2,
            y: 2,
          },
          {
            x: 1,
            y: 2,
          },
          {
            x: 1,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 0,
            y: 0,
          },
        ],
        [
          {
            x: 0,
            y: 2,
          },
          {
            x: 0,
            y: 1,
          },
          {
            x: 1,
            y: 1,
          },
          {
            x: 2,
            y: 1,
          },
          {
            x: 2,
            y: 0,
          },
        ],
      ],
    },
    {
      view: 0.5,
      theme: 6,
      frequency: 0.025,
      coords: [
        [
          {
            x: 0,
            y: 2,
          },
          {
            x: 1,
            y: 2,
          },
          {
            x: 1,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
        ],
        [
          {
            x: 2,
            y: 2,
          },
          {
            x: 2,
            y: 1,
          },
          {
            x: 1,
            y: 1,
          },
          {
            x: 0,
            y: 1,
          },
          {
            x: 0,
            y: 0,
          },
        ],
      ],
    },
    {
      view: 0.5,
      theme: 7,
      frequency: 0.025,
      coords: [
        [
          {
            x: 0,
            y: 1,
          },
          {
            x: 2,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 0,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
        ],
        [
          {
            x: 1,
            y: 1,
          },
          {
            x: 2,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 1,
            y: -1,
          },
          {
            x: 2,
            y: -1,
          },
        ],
        [
          {
            x: 1,
            y: 0,
          },
          {
            x: 0,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
          {
            x: 0,
            y: -1,
          },
          {
            x: 2,
            y: -1,
          },
        ],
        [
          {
            x: 1,
            y: 1,
          },
          {
            x: 0,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 1,
            y: -1,
          },
          {
            x: 0,
            y: -1,
          },
        ],
      ],
    },
    {
      view: 0.5,
      theme: 7,
      frequency: 0.025,
      coords: [
        [
          {
            x: 1,
            y: 2,
          },
          {
            x: 0,
            y: 2,
          },
          {
            x: 2,
            y: 2,
          },
          {
            x: 1,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
        ],
        [
          {
            x: 2,
            y: 2,
          },
          {
            x: 1,
            y: 1,
          },
          {
            x: 2,
            y: 1,
          },
          {
            x: 0,
            y: 1,
          },
          {
            x: 2,
            y: 0,
          },
        ],
        [
          {
            x: 1,
            y: 2,
          },
          {
            x: 1,
            y: 1,
          },
          {
            x: 1,
            y: 0,
          },
          {
            x: 0,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
        ],
        [
          {
            x: 0,
            y: 2,
          },
          {
            x: 1,
            y: 1,
          },
          {
            x: 0,
            y: 1,
          },
          {
            x: 2,
            y: 1,
          },
          {
            x: 0,
            y: 0,
          },
        ],
      ],
    },
    {
      view: 0.5,
      theme: 7,
      frequency: 0.05,
      coords: [
        [
          {
            x: 1,
            y: 0,
          },
        ],
      ],
    },
    {
      view: 0.5,
      theme: 7,
      frequency: 0.01,
      coords: [
        [
          {
            x: 1,
            y: 0,
          },
          {
            x: 0,
            y: 0,
          },
          {
            x: 14,
            y: 0,
          },
          {
            x: 2,
            y: 0,
          },
          {
            x: 13,
            y: 0,
          },
          {
            x: 3,
            y: 0,
          },
          {
            x: 12,
            y: 0,
          },
          {
            x: 4,
            y: 0,
          },
          {
            x: 11,
            y: 0,
          },
          {
            x: 5,
            y: 0,
          },
          {
            x: 10,
            y: 0,
          },
          {
            x: 6,
            y: 0,
          },
          {
            x: 9,
            y: 0,
          },
          {
            x: 7,
            y: 0,
          },
          {
            x: 8,
            y: 0,
          },
        ],
      ],
    },
  ];
  var o = 0;
  for (var G = 0; G < F.length; G++) {
    o += F[G].frequency;
  }

  function y() {
    var M = Math.random() * o;
    var L = 0;
    for (var x = 0; x < F.length; x++) {
      L += F[x].frequency;
      if (M < L) {
        return x;
      }
    }
    return x - 1;
  }
  this.init = function () {
    //E = g("canvas").getContext("2d");
    canvas.translate(100, 150);
    //E.lineJoin = E.lineCap = "round";
    canvas.setGlobalAlpha(0.9);
    canvas.lineWidth = 0.7;
    //canvas.setDrawStyle("black");// "#000"
    //gui.setIcon(btn_pause,"bgimage","but_play_copy[0]");
    btn_pause.setIcon("but_play_copy[0]");
  };
  var s = 0;
  var K = 0;
  var h = 0;
  var f = 0;
  var C = 0;
  var c = 0;
  var J = null;

  this.B = function (Q, P) {
    var M = new Date().getTime();
    var x = M - K;
    x = Math.max(0, Math.min(1000, x));
    A.time += x;
    if (A.mode == 2 && A.time > 181 * 1000) {
      Control.gameOver();
    } else {
      var O = A.mode == 2 ? 181 * 1000 - A.time : A.time;
      //g("time").innerHTML = niceTime(O)
    }
    var L = A.mode == 3 ? 1000 : (20 + 2980 * Math.exp(-Game.lines / 35)) >> 0;
    if (u.down) {
      Game.score += x / 100;
      if (A.mode != 3) {
        Widgets.getWidget("score").text = "score:" + Math.floor(Game.score);
      }
      L = Math.min(L, 30);
    }
    n -= Math.max(0, Math.min(1, x / L));
    K = M;
    if (C == 1) {
      x = (A.time - c) / Math.sqrt(J.length);
      if (x > 300 || x < 0) {
        C = 0;
        l();
        return;
      } else {
        Game.drawCylinder(false, false, x / 300, J);
        return;
      }
    }
    if (P || (u.left ^ u.right && M - h > 150)) {
      h = M;
      Game.move(
        P == "left" ? -1 : P == "right" ? 1 : u.left ? -1 : u.right ? 1 : 0
      );
    }
    slotY = Math.floor(n);
    if (p.length == 1) {
      for (var R = slotY; R >= 0; R--) {
        if (!b[(((v + p[0].x) % 15) + 15) % 15][R]) {
          break;
        }
      }
      if (R < 0) {
        q(slotY + 1);
        return;
      }
    } else {
      for (var N = p.length - 1; N >= 0; N--) {
        if (
          slotY + p[N].y < 0 ||
          b[(((v + p[N].x) % 15) + 15) % 15][slotY + p[N].y]
        ) {
          q(slotY + 1);
          return;
        }
      }
    }
    Game.drawCylinder(true);
  };
  this.setMode = function (i) {
    A.mode = i;
    A.clear();
    canvas.strokeText("Press Play button to play!", -50, 50);
  };
  this.start = function () {
    this.started = true;
    D = y();
    A.prepare();
    A.pause();
    A.resume();
    A.drawCylinder();
    A.time = 0;
    //gui.setIcon(btn_pause,"bgimage","but_pause_copy[0]");
    btn_pause.setIcon("but_pause_copy[0]");
  };
  this.pause = function () {
    if (Game.paused) {
      return;
    }
    Game.paused = true;
    //document.removeEventListener("keyup", t, false);
    //document.removeEventListener("keydown", z, false);
    App.clearInterval(s);
    s = 0;
    //gui.setIcon(btn_pause,"bgimage","but_resume_copy[0]");
    btn_pause.setIcon("but_resume_copy[0]");
    //  console.println("paused");
    // canvas.setFillStyle("red");
    canvas.drawImage(new Image("paused"), -50, 50);
  };
  this.resume = function () {
    if (!Game.paused) {
      //return
    }
    Game.paused = false;
    //document.addEventListener("keyup", t, false);
    //document.addEventListener("keydown", z, false);
    u.left = false;
    u.right = false;
    u.down = false;
    K = new Date().getTime();
    s = App.setInterval(updateGame, 32);
    //gui.setIcon(btn_pause,"bgimage","but_pause_copy[0]");
    btn_pause.setIcon("but_pause_copy[0]");
  };
  this.gameOver = function () {
    this.started = false;
    A.pause();
    canvas.setFont("default-large");
    //canvas.setDrawStyle("red");
    canvas.strokeText("Game Over !!", -50, 50);
    //gui.setIcon(btn_pause,"bgimage","but_play_copy[0]");
    btn_pause.setIcon("but_play_copy[0]");
  };
  var u = {
    down: false,
  };

  this.keyUp = function (keyCode) {
    switch (keyCode) {
      case 65:
      case 37:
        u.left = false;
        break;
      case 68:
      case 39:
        u.right = false;
        break;
      case 83:
      case 40:
        u.down = false;
        break;
    }
    //i.preventDefault();
    return false;
  };

  this.keyDown = function (keyCode) {
    switch (keyCode) {
      case 65:
      case 37:
        if (!u.left) {
          u.left = true;
          Game.B(null, "left");
        }
        break;
      case 68:
      case 39:
        if (!u.right) {
          u.right = true;
          Game.B(null, "right");
        }
        break;
      case 83:
      case 40:
        if (u.down === false) {
          u.down = true;
          Game.B();
        }
        break;
      case 87:
      case 38:
        Game.rotate(+1);
        break;
      case 32:
        Game.rotate(-1);
        break;
    }
    //i.preventDefault();
    return false;
  };

  function r(M, x, N) {
    if (M < 20) {
      M = F[M].theme;
    }
    var i = N || 0;
    switch (M) {
      case 0:
        return (
          "rgb(" +
          ((x * 255) >> 0) +
          "," +
          ((x * i) >> 0) +
          "," +
          ((x * i) >> 0) +
          ")"
        );
      case 1:
        return (
          "rgb(" +
          ((x * i) >> 0) +
          "," +
          ((x * 255) >> 0) +
          "," +
          ((x * i) >> 0) +
          ")"
        );
      case 2:
        return (
          "rgb(" +
          ((x * i) >> 0) +
          "," +
          ((x * i) >> 0) +
          "," +
          ((x * 255) >> 0) +
          ")"
        );
      case 3:
        return (
          "rgb(" +
          ((x * 255) >> 0) +
          "," +
          ((x * 255) >> 0) +
          "," +
          ((x * i) >> 0) +
          ")"
        );
      case 4:
        return (
          "rgb(" +
          ((x * 255) >> 0) +
          "," +
          ((x * i) >> 0) +
          "," +
          ((x * 255) >> 0) +
          ")"
        );
      case 5:
        return (
          "rgb(" +
          ((x * i) >> 0) +
          "," +
          ((x * 255) >> 0) +
          "," +
          ((x * 255) >> 0) +
          ")"
        );
      case 6:
        return (
          "rgb(" +
          ((x * 220) >> 0) +
          "," +
          ((x * 220) >> 0) +
          "," +
          ((x * 220) >> 0) +
          ")"
        );
      case 20:
        return (
          "rgb(" +
          ((x * 100) >> 0) +
          "," +
          ((x * 100) >> 0) +
          "," +
          ((x * 100) >> 0) +
          ")"
        );
      case 30:
        return (
          "rgb(" +
          ((x * 190) >> 0) +
          "," +
          ((x * 190) >> 0) +
          "," +
          ((x * 0) >> 0) +
          ")"
        );
      default:
        return (
          "rgb(" +
          ((x * 220) >> 0) +
          "," +
          ((x * 220) >> 0) +
          "," +
          ((x * 220) >> 0) +
          ")"
        );
    }
    return "black";
  }
  this.drawPiece = function () {
    var x = ((a.getPosition() % 15) + 15) % 15;
    var T = ((a._target % 15) + 15) % 15;
    var W = true; //Control.config.ghost;
    var V = Math.floor(n);
    var M = Math.sin(new Date().getTime() / 150) * 0.05 + 0.95;
    //E.fillStyle = r(d, M, 140);
    canvas.setFillStyle(r(d, M, 140));
    var U;
    var S;
    var Q;
    var L;
    var R = A.time > f ? Math.min(1, (A.time - f) / 250) : 1;
    if (W) {
      var N = w();
      if (Game.innerRadius == 0) {
        Game.innerRadius = 41;
      }
      //E.globalAlpha = 0.4 * R;
      canvas.setGlobalAlpha(0.4 * R);
      var O;
      for (P = 0; P < p.length; P++) {
        S = L = false;
        for (O = p.length - 1; O >= 0; O--) {
          if (O != P) {
            if (p[O].x == p[P].x && p[O].y == p[P].y - 1) {
              S = true;
            } else {
              if (p[O].y == p[P].y && p[O].x == p[P].x + 1) {
                L = true;
              }
            }
          }
        }
        Game.drawFront(v + p[P].x, N + p[P].y, false, x);
        Game.drawTop(v + p[P].x, N + p[P].y, x);
        Game.drawTop(v + p[P].x, N + p[P].y, x);
        Game.drawSide(v + p[P].x, N + p[P].y, x);
        if (!L) {
          Game.drawSide(v + p[P].x + 1, N + p[P].y, x);
        }
        if (!S) {
          Game.drawTop(v + p[P].x, N + p[P].y - 1, x);
        }
      }
      if (Game.innerRadius == 41) {
        Game.innerRadius = 0;
      }
    }
    canvas.setGlobalAlpha(0.9 * R);
    canvas.fillStyle = r(d, M);
    var O;
    for (var P = p.length - 1; P >= 0; P--) {
      U = S = Q = L = false;
      for (O = p.length - 1; O >= 0; O--) {
        if (O != P) {
          if (p[O].x == p[P].x) {
            if (p[O].y == p[P].y + 1) {
              U = true;
            } else {
              if (p[O].y == p[P].y - 1) {
                S = true;
              }
            }
          } else {
            if (p[O].y == p[P].y) {
              if (p[O].x == (p[P].x + 14) % 15) {
                Q = true;
              } else {
                if (p[O].x == (p[P].x + 1) % 15) {
                  L = true;
                }
              }
            }
          }
        }
      }
      Game.drawFront(v + p[P].x, n + p[P].y, false, T, Q, L, U, S);
      if (
        p[P].x + F[d].view < 1 &&
        !b[(((v + p[P].x + 1) % 15) + 15) % 15][n + p[P].y]
      ) {
        for (O = p.length - 1; O >= 0; O--) {
          if (p[O].x == 1 && p[O].y == p[P].y) {
            break;
          }
        }
        if (O < 0) {
          Game.drawSide(v + p[P].x + 1, n + p[P].y, T, U, S);
        }
      }
      if (
        p[P].x + F[d].view > 2 &&
        !b[(((v + p[P].x - 1) % 15) + 15) % 15][V + p[P].y]
      ) {
        for (O = p.length - 1; O >= 0; O--) {
          if (p[O].x == p[P].x - 1 && p[O].y == p[P].y) {
            break;
          }
        }
        if (O < 0) {
          Game.drawSide(v + p[P].x, n + p[P].y, T, U, S);
        }
      }
      if (1 || !b[(((v + p[P].x) % 15) + 15) % 15][V + p[P].y + 1]) {
        for (O = p.length - 1; O >= 0; O--) {
          if (p[O].x == p[P].x && p[O].y - 1 == p[P].y) {
            break;
          }
        }
        if (O < 0) {
          Game.drawTop(v + p[P].x, n + p[P].y, T, Q, L);
        }
      }
    }
  };
  this.drawCylinder = function (Q, N, P, ad) {
    var R = ((a.getPosition() % 15) + 15) % 15;
    canvas.setFillStyle(1, 0, 0, 1);
    canvas.fillRect(-100, -150, 200, 400); //clearrect
    canvas.reset();
    canvas.clear();
    canvas.translate(100, 150);
    //canvas.drawImage(new Image("base0"), -91, 170);
    var M = ((R % 15) + 15) % 15 >> 0;
    var T = 0;
    var S;
    var aa;
    var Z;
    var X = 0;
    var U = 0;
    var O;
    var L = new Date();
    var ae;
    var V;
    var Y = 0;
    var W = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (aa = 0; aa < 15; aa++) {
      T = [7, 8, 6, 9, 5, 10, 4, 11, 3, 12, 2, 13, 1, 14, 0][aa];
      Y = 0;
      for (S = 0; S < 15; S++) {
        U = (T + M) % 15;
        if (b[U][S]) {
          W[U] = S;
          O = 0.6 + 0.4 * Math.cos((2 * Math.PI * (U - R)) / 15);
          var ae = b[U][S][0] - 1;
          var ac = F[ae].theme;
          if (A.mode == 3 && S >= 3) {
            ac = 20;
          }
          if (A.displayGold) {
            ac = 30;
          }
          if (P && ad.contains(S)) {
            Y++;
            canvas.setGlobalAlpha(Math.max(0, Math.min(1, O * (1 - P * 1.5))));
            X = S;
          } else {
            if (A.mode == 3) {
              canvas.setGlobalAlpha(Math.max((O - 0.3) / 0.7, 0));
            } else {
              canvas.setGlobalAlpha(0.9);
            }
            X = S - Y * Math.pow(Math.max(0, P || 0), 5);
          }
          O *=
            1 +
            (Math.sin(U * 77777 + L / 871) *
              Math.sin(U * 31247 + L / 1713) *
              Math.sin(S * 41996 + L / 1713) *
              Math.sin(S * 85555 + L / 797)) /
              3;
          O = O < 0 ? 0 : O > 1 ? 1 : O;
          switch (ac) {
            case 0:
              canvas.setFillStyle(
                "rgb(" +
                  ((O * 255) >> 0) +
                  "," +
                  ((O * 0) >> 0) +
                  "," +
                  ((O * 0) >> 0) +
                  ")"
              );
              break;
            case 1:
              canvas.setFillStyle(
                "rgb(" +
                  ((O * 0) >> 0) +
                  "," +
                  ((O * 255) >> 0) +
                  "," +
                  ((O * 0) >> 0) +
                  ")"
              );
              break;
            case 2:
              canvas.setFillStyle(
                "rgb(" +
                  ((O * 0) >> 0) +
                  "," +
                  ((O * 0) >> 0) +
                  "," +
                  ((O * 255) >> 0) +
                  ")"
              );
              break;
            case 3:
              canvas.setFillStyle(
                "rgb(" +
                  ((O * 255) >> 0) +
                  "," +
                  ((O * 255) >> 0) +
                  "," +
                  ((O * 0) >> 0) +
                  ")"
              );
              break;
            case 4:
              canvas.setFillStyle(
                "rgb(" +
                  ((O * 255) >> 0) +
                  "," +
                  ((O * 0) >> 0) +
                  "," +
                  ((O * 255) >> 0) +
                  ")"
              );
              break;
            case 5:
              canvas.setFillStyle(
                "rgb(" +
                  ((O * 0) >> 0) +
                  "," +
                  ((O * 255) >> 0) +
                  "," +
                  ((O * 255) >> 0) +
                  ")"
              );
              break;
            case 6:
              canvas.setFillStyle(
                "rgb(" +
                  ((O * 220) >> 0) +
                  "," +
                  ((O * 220) >> 0) +
                  "," +
                  ((O * 220) >> 0) +
                  ")"
              );
              break;
            case 20:
              canvas.setFillStyle(
                "rgb(" +
                  ((O * 100) >> 0) +
                  "," +
                  ((O * 100) >> 0) +
                  "," +
                  ((O * 100) >> 0) +
                  ")"
              );
              break;
            case 30:
              canvas.setFillStyle(
                "rgb(" +
                  ((O * 190) >> 0) +
                  "," +
                  ((O * 190) >> 0) +
                  "," +
                  ((O * 0) >> 0) +
                  ")"
              );
              break;
            default:
              canvas.setFillStyle(
                "rgb(" +
                  ((O * 220) >> 0) +
                  "," +
                  ((O * 220) >> 0) +
                  "," +
                  ((O * 220) >> 0) +
                  ")"
              );
              break;
          }
          var ab = (((U - R) % 15) + 15) % 15;
          Game.drawFront(
            U,
            X,
            ab > 3.3 && ab < 10.7,
            R,
            (ab > 11.9 || ab < 10.9) &&
              (V = b[(U + 14) % 15][S]) &&
              V[1] == b[U][S][1],
            (ab < 2.1 || ab > 3.1) &&
              (V = b[(U + 1) % 15][S]) &&
              V[1] == b[U][S][1],
            (V = b[U][S + 1]) && V[1] == b[U][S][1],
            (V = b[U][S - 1]) && V[1] == b[U][S][1]
          );
          if (!b[U][S + 1] || (V = ad && ad.contains(S + 1))) {
            Game.drawTop(
              U,
              X,
              R,
              (V || !b[(U + 14) % 15][S + 1]) &&
                (V = b[(U + 14) % 15][S]) &&
                V[1] == b[U][S][1],
              (V || !b[(U + 1) % 15][S + 1]) &&
                (V = b[(U + 1) % 15][S]) &&
                V[1] == b[U][S][1]
            );
          }
          if (ab > 6.5 && ab < 14) {
            if (!b[(((T + 1 + M) % 15) + 15) % 15][S]) {
              Game.drawSide(
                U + 1,
                X,
                R,
                (V = b[U][S + 1]) && V[1] == b[U][S][1],
                (V = b[U][S - 1]) && V[1] == b[U][S][1]
              );
            }
          }
          if (ab < 7.5) {
            if (!b[(((T - 1 + M) % 15) + 15) % 15][S]) {
              Game.drawSide(
                U,
                X,
                R,
                (V = b[U][S + 1]) && V[1] == b[U][S][1],
                (V = b[U][S - 1]) && V[1] == b[U][S][1]
              );
            }
          }
        }
      }
    }
    for (aa = 0; aa < 15 && !N; aa++) {
      O = W[aa];
      for (Z = 1; Z < 5; Z++) {
        O = Math.max(O, W[(aa + Z) % 15] - Z, W[(aa - Z + 15) % 15] - Z);
      }
      if (O > 9) {
        canvas.setGlobalAlpha((O - 9) / 5);
        canvas.setFillStyle(
          "rgb(" +
            ((255 * (Math.sin(new Date() / 200) / 2 + 1 / 2)) >> 0) +
            ",30,0)"
        );
        Game.drawTop(aa, 14, R, null, null, true);
      }
    }
    if (Q) {
      Game.drawPiece();
    }
    renderScore();
  };
  this.clear = function () {
    var L, x, M;
    v = 0;
    b = [];
    for (L = 0; L < 15; L++) {
      b[L] = new Array(15);
    }
    a.reset();
    Game.lines = 0;
    Game.score = 0;
    //g("score").innerHTML = (A.mode == 3) ? "+6" : "0";
    //Widgets.getWidget("score").text=((A.mode == 3) ? "+6" : "0");
    if (A.mode == 1) {
      var N = b;
      N[10][0] = N[11][0] = N[10][1] = N[11][1] = [1, 1];
      N[9][0] = N[9][1] = N[9][2] = N[9][3] = [2, 2];
      N[8][0] = N[7][0] = N[7][1] = N[6][0] = [3, 3];
      N[12][0] = N[13][0] = N[13][1] = N[13][2] = [4, 4];
      N[3][0] = N[4][0] = N[5][0] = N[5][1] = [5, 5];
      N[1][0] = N[2][0] = N[2][1] = N[3][1] = [6, 6];
      N[7][3] = N[7][2] = N[6][2] = N[6][1] = [7, 7];
      k = 8;
    }
    if (A.mode == 2) {
      var N = b;
      N[0][0] =
        N[3][0] =
        N[6][0] =
        N[9][0] =
        N[12][0] =
          [(Math.random() * 6 + 1) >> 0, 0];
      k = 1;
    }
    if (A.mode == 3) {
      for (L = 0; L < 9; L++) {
        for (x = 0; x < 5; x++) {
          M = (Math.random() * 15) >> 0;
          if (b[M][L]) {
            x--;
            continue;
          } else {
            b[M][L] = [1, 0];
          }
        }
      }
      k = 1;
    }
  };
  this.change = function () {
    d = D;
    D = y();
    nextblock = "blocks[" + F[D].theme + "]";
    nextblockImg.src = nextblock;
    //g("next").style.backgroundPosition = (-F[D].theme * 80) + "px 0";
    e = 0;
    n = 16;
    a.setTarget(a._target - F[d].view);
    p = F[d].coords[0];
    f = A.time;
  };

  function w() {
    slotY = Math.floor(n);
    var x;
    var L;
    if (p.length == 1) {
      for (var L = 0; ; L++) {
        if (!b[(((v + p[0].x) % 15) + 15) % 15][L]) {
          return L;
        }
      }
    }
    for (x = 0; ; x++) {
      for (L = p.length - 1; L >= 0; L--) {
        if (
          slotY + p[L].y - x < 0 ||
          b[(((v + p[L].x) % 15) + 15) % 15][slotY + p[L].y - x]
        ) {
          return slotY - x + 1;
        }
      }
    }
  }

  function q(N) {
    u.down = null;
    lastDrop = new Date().getTime();
    for (var M = p.length - 1; M >= 0; M--) {
      if (N + p[M].y > 14) {
        A.drawCylinder(false, true);
        Control.gameOver();
        return;
      }
    }
    for (var M = p.length - 1; M >= 0; M--) {
      b[(((v + p[M].x) % 15) + 15) % 15][N + p[M].y] = [d + 1, k];
    }
    ++k;
    var x = [];
    for (var L = 0; L < 15; L++) {
      for (M = 0; M < 15; M++) {
        if (!b[M][L]) {
          break;
        }
      }
      if (M == 15) {
        for (M = 0; M < 15; M++) {
          b[M][L] = [b[M][L][0], 0];
        }
        x.push(L);
      }
    }
    if (x.length) {
      C = 1;
      c = A.time;
      J = x;
    } else {
      l();
    }
  }

  function l() {
    var i = 0;
    for (var x = 0; x < 15; x++) {
      for (G = 0; G < 15; G++) {
        if (!b[G][x]) {
          break;
        }
      }
      if (G == 15) {
        for (G = 0; G < 15; G++) {
          b[G].splice(x, 1);
        }
        x--;
        Game.lines++;
        i++;
      }
    }
    Game.score += [0, 100, 250, 400, 600, 1000][i];
    if (A.mode == 3) {
      for (x = 14; x > 2; x--) {
        for (G = 0; G < 15; G++) {
          if (b[G][x]) {
            break;
          }
        }
        if (G < 15) {
          break;
        }
      }
      //g("score").innerHTML = "+" + (x - 2);
      Widgets.getWidget("score").text = "+" + (x - 2); //
      if (x == 2) {
        Game.change();
        Game.drawCylinder();
        Control.gameOver(true);
        a.setTarget(a._target + F[d].view);
        return;
      }
    } else {
      // g("score").innerHTML = Math.floor(Game.score)
      Widgets.getWidget("score").text = Math.floor(Game.score);
    }
    a.setTarget(a._target + F[d].view);
    Game.change();
  }
  this.prepare = function () {
    v = -2;
    this.change();
  };

  function I() {
    var N, M;
    var L = Math.floor(n);
    var x;
    if (p.length == 1) {
      for (M = L; M >= 0; M--) {
        if (!b[(((v + p[0].x) % 15) + 15) % 15][M]) {
          return false;
        }
      }
      return true;
    }
    for (x = p.length - 1; x >= 0; x--) {
      N = (((v + p[x].x) % 15) + 15) % 15;
      M = L + p[x].y;
      if (M < 0 || b[N][M]) {
        return true;
      }
    }
    if (n % 1 > 1 / 2) {
      L = Math.ceil(n);
      for (x = p.length - 1; x >= 0; x--) {
        N = (((v + p[x].x) % 15) + 15) % 15;
        M = L + p[x].y;
        if (M < 0 || b[N][M]) {
          return true;
        }
      }
    }
    return false;
  }
  this.move = function (i) {
    v += i;
    if (I()) {
      v -= i;
      return false;
    }
    a.setTarget(a._target + i);
    return true;
  };
  this.rotate = function (i, x) {
    if (d == 0) {
      return false;
    }
    e = (e + i + 4) % F[d].coords.length;
    p = F[d].coords[e];
    if (x) {
      return false;
    }
    if (I() && !this.move(1) && !this.move(-1)) {
      this.rotate(-i, true);
      return false;
    }
    return true;
  };
})();
(function () {
  var a = [];
  Function.prototype.wait = function () {
    var b = a.length;
    while (b-- > 0) {
      if (a[b][0] == this) {
        return;
      }
    }
    b = arguments;
    a[a.length] = [
      this,
      App.setTimeout(function () {
        a.shift()[0].apply(this, b);
      }, 1),
    ];
  };
})();
Array.prototype.contains = function (b) {
  for (var a = this.length - 1; a >= 0; a--) {
    if (this[a] === b) {
      return true;
    }
  }
  return false;
};
Number.prototype.toLength = function (b) {
  var a = this.toString();
  while (a.length < b) {
    a = "0" + a;
  }
  return a;
};

function niceTime(a) {
  a = Game.mode == 2 ? 181 * 1000 - a : a;
  a /= 1000;
  a >>= 0;
  return ((a / 60) >> 0) + ":" + (a % 60).toLength(2);
}

function g(a) {
  //return document.getElementById(a)
}

function Control() {
  this.config = {
    ghost: true,
    skin: 0,
  };

  function d() {
    var h;
    /*
for (var f = 0; f < 3; f++) {
    for (h = 0; h < 3; h++) {
        setCookie("best" + f + "" + h + "score", c[f][h][0]);
        setCookie("best" + f + "" + h + "name", c[f][h][1])
    }
}
*/
  }

  function e(f) {
    str = "<B>" + ["Traditional", "Time Attack", "Garbage"][f] + "</B>";
    for (j = 0; j < 3; j++) {
      str +=
        "<BR>" +
        (j + 1) +
        ". " +
        c[f][j][1] +
        " (" +
        (f == 2 ? niceTime(c[f][j][0]) : c[f][j][0]) +
        ")";
    }
    return str;
  }

  function b() {
    var k, f;
    for (var h = 0; h < 3; h++) {
      //g("best" + (h + 1)).innerHTML = e(h)
    }
  }
  var c = [
    [
      ["0", "Empty"],
      ["0", "Empty"],
      ["0", "Empty"],
    ],
    [
      ["0", "Empty"],
      ["0", "Empty"],
      ["0", "Empty"],
    ],
    [
      ["3599000", "Empty"],
      ["3599000", "Empty"],
      ["3599000", "Empty"],
    ],
  ];
  /*
if (getCookie("best11score") == null) {
d()
} else {
for (var a = 0; a < 3; a++) {
    for (pos = 0; pos < 3; pos++) {
        c[a][pos][0] = Number(getCookie("best" + a + "" + pos + "score")) || 0;
        c[a][pos][1] = getCookie("best" + a + "" + pos + "name")
    }
}
}
*/
  b();
  this.gameOver = function (f) {
    Game.gameOver();
    var l = Math.floor(Game.score);
    var k = Game.time;
    if (f === false) {
      return;
    }
    var h = "Your score of " + l + " did not achieve a high score.";
    var i = false;
    if (Game.mode == 1) {
      if (l > c[0][2][0]) {
        i = true;
      }
    }
    if (Game.mode == 2) {
      if (k > 181 * 1000) {
        if (l > c[1][2][0]) {
          i = true;
        }
      } else {
        h =
          "Failure. You must survive for 3 minutes to qualify for a high score in <I>Time Attack</I>.";
      }
    }
    if (Game.mode == 3) {
      if (!f) {
        h =
          "Failure. You must clear all but three rows to qualify for a high score in <I>Garbage</I>.";
      } else {
        if (k < c[2][2][0]) {
          i = true;
        } else {
          h = "Your time of " + niceTime(k) + " did not achieve a high score.";
        }
      }
    }
    UI.gameOver();
    /*
g("winner").style.display = i ? "block" : "none";
g("newgame").style.display = i ? "none" : "block";
g("sorryText").innerHTML = i ? "" : h;
g("gameover").style.display = "block"
*/
  };
  this.close = function () {
    //window.close()
  };
  this.restartGame = function () {
    Game.gameOver(false);
    Control.startGame(Game.mode);
  };
  this.startGame = function (f) {
    UI.setGameMode(f || 1);
    Game.setMode(f);
    Game.start();
  };
  this.pauseGame = function () {
    Game.pause();
    UI.pauseGame();
  };
  this.resumeGame = function () {
    Game.resume();
    UI.resumeGame();
  };
  /*
g("high_form").onsubmit = function() {
var k = g("high_name").value;
if (!k) {
    return
}
g("winner").style.display = "none";
g("newgame").style.display = "block";
var h = Game.mode < 3 ? Math.floor(Game.score) : Game.time;
for (var f = 2; f >= 0; f--) {
    if (Game.mode < 3 && h > c[Game.mode - 1][f][0] || Game.mode == 3 && h < c[2][f][0]) {
        if (f < 2) {
            c[Game.mode - 1][f + 1][0] = c[Game.mode - 1][f][0];
            c[Game.mode - 1][f + 1][1] = c[Game.mode - 1][f][1]
        }
    } else {
        break
    }
}
f++;
c[Game.mode - 1][f][0] = h;
c[Game.mode - 1][f][1] = k;
d();
b();
g("sorryText").innerHTML = '<div style="padding-left:60px;">' + e(Game.mode - 1) + "</div>";
return false
};
document.getElementsByTagName("body")[0].style.visibility = "visible"
*/
}

function renderScore() {
  canvas_score.clear();
  canvas_score.setFont("score_font");
  canvas_score.setFillStyle("white");
  canvas_score.fillText(Math.floor(Game.score), 22, 58);
  canvas_score.fillText(niceTime(Game.time), 22, 112);
  canvas_score.drawImage(
    nextblockImg,
    canvas_score.canvas.width / 2 - nextblockImg.width / 2,
    145
  );
}

function startGame(mode) {
  Control.startGame(mode);
}
/**
 * Runs when widget is clicked
 */
function but_pause_onClick() {
  Game.paused ? Control.resumeGame() : Control.pauseGame();
}

Game.gameOver();
Game.init();
Control = new Control();
UI.init();

/**
 * Runs  when the widget recieves mouse event (events are similar to java.awt events).
 * @param x{Number} : x coordinate of mouse in widget space (top left as origin)
 * @param y{Number} : y coordinate of mouse in widget space (top left as origin)
 * @param clickCount{Number} :  Number of clicks
 * @param id{Number}: EventType: 500(clicked),501(pressed),502(released),503(moved),504(entered),505(exited),506(dragged)
 * @param button{Number} : Button involved in event {1=Left, 2=Middle and 3=Right mouse button*/
function canvas_onMouseEvent(x, y, clickCount, id, button) {}
/**
 * Runs when widget recieves key event (events are similar to java.awt events).
 * @param keyChar{char} : the character(if any) associated with keyevent
 * @param keyCode{Number} : key code {@see com.jogamp.newt.event.KeyEvent} for keycodes
 * @param id{Number} : EventType 401(keyPressed), 402(KeyReleased)
 * Note that key events are received only when widget is active (click on widget to activate)*/
function canvas_onKeyEvent(keyChar, keyCode, key, id) {
  if (id == 401) {
    Game.keyDown(keyCode);
  } else {
    Game.keyUp(keyCode);
  }
}

function updateGame() {
  if (Game.paused) {
    //Game.drawCylinder();
    return;
  }
  Game.B();
}
/**
 * Runs when widget is clicked
 */
function btn_pause_onClick() {
  if (!Game.started) {
    Control.startGame(1);
    return;
  }
  Game.paused ? Control.resumeGame() : Control.pauseGame();
  return;
}
