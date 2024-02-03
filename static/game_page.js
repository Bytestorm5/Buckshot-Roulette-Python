// Function to fetch data from the /data endpoint
var prev_data = undefined
var data = undefined
var executing = false
function fetchData() {
    fetch('/data')
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(new_data => {
        data = new_data
        if (prev_data == undefined || prev_data['turn_id'] != new_data['turn_id']) {
            prev_data = new_data
            update(new_data)
            executing = true
        }
        else if (new_data['lock'] && !executing) {
            unlockEndpoint()
        }        
      })
      .catch(error => {
        console.error('There has been a problem with your fetch operation:', error);
      });
  }
  
  // Function to call the /unlock endpoint
  function unlockEndpoint() {
    fetch('/unlock', { method: 'GET' }) // Assuming POST, but this could be GET or another HTTP method depending on the API
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        console.log('Endpoint /unlock called successfully.');
      })
      .catch(error => {
        console.error('There has been a problem with your fetch operation:', error);
      });
  }

  function actionEndpoint(action) {
    fetch(`/action?action=${action}`, { method: 'GET' }) // Assuming POST, but this could be GET or another HTTP method depending on the API
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        console.log('Endpoint /action called successfully.');        
      })
      .catch(error => {
        console.error('There has been a problem with your fetch operation:', error);
      });
  }

  function shoot(by, at) {
        executing = true
        if (data.turn != 0) {
            return
        }
        const shotgun_img = document.querySelector('#shotgun_img');
        shotgun_img.style.transition = 'transform 2s';

        if (at == 0) {
            shotgun_img.style.transform = 'rotate(90deg)';
            if (by == 0) {
                actionEndpoint('self');
            }            
            setTimeout(() => {
                shotgun_img.style.transform = 'rotate(0deg)';                
            }, 3000); // Delay of 2 seconds (2000 milliseconds)
            setTimeout(() => {
                unlockEndpoint()
                executing = false   
            }, 5000)
        }
        else if (at == 1) {
            shotgun_img.style.transform = 'rotate(-90deg)';
            if (by == 0) {
                actionEndpoint('op');
            }    
            setTimeout(() => {
                shotgun_img.style.transform = 'rotate(0deg)';
            }, 3000); // Delay of 2 seconds
            setTimeout(() => {
                unlockEndpoint()
                executing = false   
            }, 5000)
        }   
        
    }
    var hide = true
    function toggleHide() {
        hide = !hide
        update()
    }

    function constructItemArray(obj) {
        let result = new Array(8).fill(undefined);
        let index = 0;
    
        for (const [key, value] of Object.entries(obj)) {
            for (let i = 0; i < value; i++) {
                if (index < 8) {
                    result[index++] = key;
                }
            }
        }
    
        return result;
    }

  // Function to update with the data
  function update() {
    basic_info = document.getElementsByClassName('basic_info')[0]
    basic_info.innerHTML = `
    Turn: ${data.turn == 0 ? "You" : "Opponent"}<br>
    Lock: ${data.lock}<br>
    Active Items:<br>
     - Handcuffs: ${data.active_items.handcuffs > 0}<br>
     - Saw: ${data.active_items.saw > 0}<br>
    Known Shell: ${data.known_shell}<br>
    Last Action: ${data.last_action}<br>
    Ratio: ${data.shotgun_info[0]} / ${data.shotgun_info[1]}<br>
    <button type="button" onclick="unlockEndpoint()">Unlock</button>
    <button type="button" onclick="toggleHide()">Toggle Hide</button>
    `

    if (data.active_items.saw <= 0) {
        const shotgun_img = document.querySelector('#shotgun_img');
        shotgun_img.src = "/static/images/shotgun.png"
    }
    else {
        const shotgun_img = document.querySelector('#shotgun_img');
        shotgun_img.src = "/static/images/shotgun_sawed.png"
    }


    // Lives
    charge_width = 225 / data.max_charges
    p_area = document.getElementsByClassName('player_area')[0]
    p_area.innerHTML = ""
    for (let i = 0; i < data.max_charges; i++) {
        if (i < data['charges'][0]) {
            p_area.innerHTML += `<img src="/static/images/charge.png" width="${charge_width}px">`
        }
        else {
            p_area.innerHTML += `<img src="/static/images/charge_blank.png" width="${charge_width}px">`
        }
    }
    op_area = document.getElementsByClassName('opponent_area')[0]
    op_area.innerHTML = ""
    for (let i = 0; i < data.max_charges; i++) {
        if (i < data['charges'][1]) {
            op_area.innerHTML += `<img src="/static/images/charge.png" width="${charge_width}px">`
        }
        else {
            op_area.innerHTML += `<img src="/static/images/charge_blank.png" width="${charge_width}px">`
        }
    }

    bullet_height = Math.min(225 / data.shotgun.length, 75)
    aspect_ratio = 2.67391304
    shotgun_info = document.getElementsByClassName('shotgun_info')[0]
    shotgun_info.innerHTML = ""
    for (let i = 0; i < data.shotgun.length && !hide; i++) {
        if (data.shotgun[i]) {
            shotgun_info.innerHTML += `<img src="/static/images/bullet_live.png" height="${bullet_height}px" width="${bullet_height * aspect_ratio}">`
        }
        else {
            shotgun_info.innerHTML += `<img src="/static/images/bullet_blank.png" height="${bullet_height}px" width="${bullet_height * aspect_ratio}">`
        }
    }

    p_items = constructItemArray(data.player_items)
    op_items = constructItemArray(data.op_items)

    p_slots = document.getElementsByClassName('p_item_slot')
    op_slots = document.getElementsByClassName('op_item_slot')

    for (let i = 0; i < 8; i++) {
        p_slots[i].innerHTML = ""
        if (p_items[i] != undefined) {
            var addons = `style='height="100%";width="100%";`
            if (data.moves.includes(p_items[i])) {
                addons = `onclick="actionEndpoint('${p_items[i]}')" style='background-color: lightgreen;height="100%";width="100%";'`
            }
            p_slots[i].innerHTML = `<button type="button" ${addons}><img src="/static/images/${p_items[i]}.png" height="100%" width="100%"></button>`
        }
        op_slots[i].innerHTML = ""
        if (op_items[i] != undefined) {
            op_slots[i].innerHTML = `<img src="/static/images/${op_items[i]}.png" height="100%" width="100%">`
        }
    }

    if (data.lock || executing) {
        // Update Last Action
        la = data.last_action
        if (la[0] == 1 && la[1] == 'op') {
            shoot(1, 0)
        }
        else if (la[0] == 1 && la[1] == 'self') {
            shoot(1, 1)
        }
        executing = false
        unlockEndpoint()
    }


    console.log(data);
    // Implement the logic of the update function using the data
  }
  
  // Interval in milliseconds (e.g., 5000ms = 5 seconds)
  const interval = 1000;
  
  // Set up an interval to call fetchData repeatedly
  setInterval(fetchData, interval);
  